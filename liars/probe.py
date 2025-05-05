import os
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
from sklearn.cluster import HDBSCAN
from liars.constants import DATA_PATH, CACHE_PATH
from tqdm import trange


class Probe(nn.Module):
    def __init__(self, model_name: str, probe_type: str, last_only: bool = True, split: float = 0.8) -> None:
        super().__init__()
        self.model_name = model_name
        self.probe_type = probe_type
        self.last_only = last_only
        self.split = split

        # === ACTIVATIONS, LOAD DATA --> LABELS ===
        activation_path = f"{CACHE_PATH}/activations/{model_name}/"
        if last_only:
            x_true = t.load(f"{activation_path}/true_probe_fitting_last.pt", weights_only=False)
            x_false = t.load(f"{activation_path}/false_probe_fitting_last.pt", weights_only=False)
            data = pd.read_json(f"{DATA_PATH}/gom_data_for_probes.jsonl", orient="records", lines=True)
            # invert labels as we want to predict lies
            data["label"] = data["label"].apply(lambda x: not x)
            labels = t.tensor(data["label"].values, dtype=t.bfloat16)
        else:
            x_true = t.load(f"{activation_path}/true_probe_fitting_full.pt", weights_only=False)
            x_false = t.load(f"{activation_path}/false_probe_fitting_full.pt", weights_only=False)
            labels = t.load(f"{activation_path}/probe_fitting_labels.pt", weights_only=False)
        assert len(labels) == len(x_true) == len(x_false)

        # train/test split now as we use standard activations for testing
        perm = t.randperm(len(x_true))
        x_true, x_false, labels = x_true[perm], x_false[perm], labels[perm]
        split = int(0.8 * len(x_true))
        x_true_train, x_true_test = x_true[:split], x_true[split:]
        x_false_train, x_false_test = x_false[:split], x_false[split:]
        self.Y_train, Y_test = labels[:split], labels[split:]
        self.Y_test = t.cat([Y_test, (~Y_test.bool()).to(t.bfloat16)], dim=0)

        # TODO: cluster-norm
        if probe_type == "ccs" or probe_type == "crc-tpc":
            self.x_true_train, self.x_false_train = self._cluster_norm(x_true_train, x_false_train)
        else:
            # centering
            self.x_true_train = x_true_train - x_true_train.mean(dim=1, keepdim=True)
            self.x_false_train = x_false_train - x_false_train.mean(dim=1, keepdim=True)
        self.X_train = self.x_true_train - self.x_false_train
        self.X_test = t.cat([x_true_test, x_false_test], dim=0)

        self.d_model = self.X_train.shape[-1]
        self.probe = nn.Linear(self.d_model, 1, dtype=t.bfloat16, bias=False)

    def forward(self, x) -> t.Tensor:
        return self.probe(x)
    
    def _cluster_norm(self, x_true_train: t.Tensor, x_false_train: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        print("applying cluster-normalization")
        # fit HDBSCAN
        clusterer = HDBSCAN()
        cluster_labels = clusterer.fit_predict((x_true_train + x_false_train).float() / 2)
        centered_x_true_train = x_true_train.clone()
        centered_x_false_train = x_false_train.clone()
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:
                # use global mean for outliers
                global_mean_true = x_true_train.mean(dim=0, keepdim=True)
                global_mean_false = x_false_train.mean(dim=0, keepdim=True)
                outlier_mask = (cluster_labels == -1)
                centered_x_true_train[outlier_mask] -= global_mean_true
                centered_x_false_train[outlier_mask] -= global_mean_false
            else:
                # center each main cluster independently
                cluster_mask = (cluster_labels == cluster_id)
                cluster_mean_true = x_true_train[cluster_mask].mean(dim=0, keepdim=True)
                cluster_mean_false = x_false_train[cluster_mask].mean(dim=0, keepdim=True)
                centered_x_true_train[cluster_mask] -= cluster_mean_true
                centered_x_false_train[cluster_mask] -= cluster_mean_false
        return x_true_train, x_false_train

    def eval(self) -> None:
        with t.no_grad():
            test_probs = F.sigmoid(self.probe(self.X_test).squeeze(-1))
            fpr, tpr, _ = metrics.roc_curve(self.Y_test.cpu().float(), test_probs.detach().cpu().float())
            test_auroc = metrics.auc(fpr, tpr)
            print(f"test AUROC: {test_auroc:.3f}")

    def fit(self, **kwargs) -> None:
        if self.probe_type == "supervised":
            self._fit_supervised(**kwargs)
        elif self.probe_type == "diff-in-means":
            self._fit_diff_in_means()
        elif self.probe_type == "crc-tpc":
            self._fit_crc_tpc()
        elif self.probe_type == "ccs":
            self._fit_ccs(**kwargs)
        else:
            raise ValueError(f"probe type {self.probe_type} not supported")

    def _fit_ccs(self, batch_size: int = 64, nepoch: int = 5) -> None:
        print(f"fitting {self.probe_type} probe")
        # batch data
        nbatch = len(self.x_true_train) // batch_size
        # prepare probe
        opt = t.optim.AdamW(self.probe.parameters(), lr=1e-3, weight_decay=1e-5)
        # train
        for epoch in range(nepoch):
            # reshuffle training data
            perm = t.randperm(len(self.X_train))
            self.x_true_train, self.x_false_train = self.x_true_train[perm], self.x_false_train[perm]
            for b in trange(nbatch, desc=f"epoch {epoch}"):
                xtt, xft = self.x_true_train[b*batch_size:(b+1)*batch_size], self.x_false_train[b*batch_size:(b+1)*batch_size]
                # forward pass
                p_xtt = F.sigmoid(self.probe(xtt).squeeze(-1))
                p_xft = F.sigmoid(self.probe(xft).squeeze(-1))
                l_inf = (t.min(p_xtt, p_xft)**2).mean()
                l_con = ((p_xtt - (1 - p_xft))**2).mean()
                L = l_inf + l_con
                # backward pass
                opt.zero_grad()
                L.backward()
                opt.step()
        # orient probe
        probe_direction = self.probe.weight.data.clone()
        with t.no_grad():
            scores = F.sigmoid(self.probe(self.X_train).squeeze(-1))
            fpr, tpr, _ = metrics.roc_curve(self.Y_train.cpu().float(), scores.detach().cpu().float())
            auroc = metrics.auc(fpr, tpr)
            # try flipping direction
            self.probe.weight.data = -probe_direction
            scores_flipped = F.sigmoid(self.probe(self.X_train).squeeze(-1))
            fpr, tpr, _ = metrics.roc_curve(self.Y_train.cpu().float(), scores_flipped.detach().cpu().float())
            auroc_flipped = metrics.auc(fpr, tpr)   
            # keep orientation with better AUROC
            if auroc_flipped < auroc:
                self.probe.weight.data = probe_direction

    def _fit_crc_tpc(self) -> None:
        print(f"fitting {self.probe_type} probe")
        # get first principal component
        U, S, V = t.pca_lowrank(self.X_train.float(), q=1)
        probe_direction = V[:, 0]
        # try both orientations and pick better one
        self.probe.weight.data = probe_direction.unsqueeze(0).to(t.bfloat16)
        with t.no_grad():
            scores = F.sigmoid(self.probe(self.X_train).squeeze(-1))
            fpr, tpr, _ = metrics.roc_curve(self.Y_train.cpu().float(), scores.detach().cpu().float())
            auroc = metrics.auc(fpr, tpr)
            # try flipping direction
            self.probe.weight.data = -probe_direction.unsqueeze(0).to(t.bfloat16)
            scores_flipped = F.sigmoid(self.probe(self.X_train).squeeze(-1))
            fpr, tpr, _ = metrics.roc_curve(self.Y_train.cpu().float(), scores_flipped.detach().cpu().float())
            auroc_flipped = metrics.auc(fpr, tpr)   
            # keep orientation with better AUROC
            if auroc_flipped < auroc:
                self.probe.weight.data = probe_direction.unsqueeze(0).to(t.bfloat16)

    def _fit_diff_in_means(self) -> None:
        print(f"fitting {self.probe_type} probe")
        # take difference between mean of lies and mean of truths
        mask = self.Y_train.bool()
        x_lie = self.X_train[mask].mean(dim=0)
        x_truth = self.X_train[~mask].mean(dim=0)
        x_diff = (x_lie - x_truth).unsqueeze(0)
        self.probe.weight.data = x_diff        
    
    def _fit_supervised(self, batch_size: int = 64, nepoch: int = 1) -> None:
        print(f"fitting {self.probe_type} probe")
        # batch data
        nbatch = len(self.X_train) // batch_size
        # prepare probe
        opt = t.optim.Adam(self.probe.parameters(), lr=1e-3)
        loss = nn.BCEWithLogitsLoss()
        # train
        for epoch in range(nepoch):
            # reshuffle training data
            perm = t.randperm(len(self.X_train))
            self.X_train, self.Y_train = self.X_train[perm], self.Y_train[perm]
            for b in trange(nbatch, desc=f"epoch {epoch}"):
                x, y = self.X_train[b*batch_size:(b+1)*batch_size], self.Y_train[b*batch_size:(b+1)*batch_size]
                # forward pass
                out = self.probe(x).squeeze(-1)
                # compute loss
                L = loss(out, y)
                # backward pass
                opt.zero_grad()
                L.backward()
                opt.step()


if __name__ == "__main__":
    models = os.listdir(f"{CACHE_PATH}/activations/")
    for model_name in models:
        print(f"fitting probes for{model_name}")
        for probe_type in ["supervised", "diff-in-means", "crc-tpc", "ccs"]:
            print("="*100)
            probe = Probe(model_name, probe_type)
            probe.fit()
            probe.eval()
            output_path = f"{CACHE_PATH}/probes/{model_name}"
            os.makedirs(output_path, exist_ok=True)
            t.save(probe.probe.weight.data, f"{output_path}/{probe_type}.pt")
        print()
        print()