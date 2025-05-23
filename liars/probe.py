import os
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from liars.constants import CACHE_PATH
from tqdm import trange


class Probe(nn.Module):
    def __init__(
            self,
            model_name: str,
            probe_type: str,
            split: float = 0.8
    ) -> None:
        super().__init__()
        
        # === LOAD ACTIVATIONS AND LABELS
        path = f"{CACHE_PATH}/activations/{model_name}"
        self.honest = t.load(f"{path}/honest.pt", weights_only=False).float()
        self.dishonest = t.load(f"{path}/dishonest.pt", weights_only=False).float()

        # === PREPARE DATA WITH LABELS ===
        X = t.cat((self.honest, self.dishonest), dim=0)
        Y = t.tensor(
            [0 for _ in range(len(self.honest))] +
            [1 for _ in range(len(self.dishonest))],
            dtype=t.float32
        )
        
        # === STRATIFIED TRAIN/TEST SPLIT ===
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=1-split, stratify=Y.cpu(), random_state=123456
        )
        
        # === Z-SCORE NORMALIZATION (ONLY ON TRAINING DATA) ===
        self.mu = X_train.mean(dim=0)
        self.sigma = X_train.std(dim=0) + 1e-8
        
        # Apply z-score normalization
        self.X_train = (X_train - self.mu) / self.sigma
        self.X_test = (X_test - self.mu) / self.sigma
        self.Y_train = Y_train
        self.Y_test = Y_test

        # init probe in float32
        d_model = self.honest.shape[1]
        self.probe = nn.Linear(d_model, 1, dtype=t.float32, bias=False)

        self.probe_type = probe_type
        self.split = split
        self.model_name = model_name

    def forward(self, x) -> t.Tensor:
        return self.probe(x)

    def fit(self, **kwargs) -> None:
        if self.probe_type == "supervised":
            self._fit_supervised(**kwargs)
        elif self.probe_type == "diff-in-means":
            self._fit_mean_difference(**kwargs)
        elif self.probe_type == "ccs":
            self._fit_ccs(**kwargs)
        elif self.probe_type == "crc-tpc":
            self._fit_crc_tpc(**kwargs)
        else:
            raise ValueError(f"probe type {self.probe_type} not supported")
    
    def _fit_supervised(self, batch_size: int = 64, nepoch: int = 5, lambda_l2: float = 10.0) -> None:
        print(f"fitting {self.probe_type} probe")
        
        # use pre-split and normalized data
        X_train, Y_train = self.X_train, self.Y_train
        X_test, Y_test = self.X_test, self.Y_test
        
        # prepare probe
        opt = t.optim.AdamW(self.probe.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        def loss(logits, labels):
            ce = criterion(logits, labels)
            l2 = lambda_l2 * self.probe.weight.pow(2).sum()
            return ce + l2
        
        # train
        for epoch in range(nepoch):
            # reshuffle training data
            perm = t.randperm(len(X_train))
            X_train_shuffled, Y_train_shuffled = X_train[perm], Y_train[perm]
            
            # process all batches including the last incomplete one
            for b in trange((len(X_train) + batch_size - 1) // batch_size, desc=f"epoch {epoch}"):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, len(X_train))
                x, y = X_train_shuffled[start_idx:end_idx], Y_train_shuffled[start_idx:end_idx]
                
                # forward pass
                out = self.probe(x).squeeze(-1)
                # compute loss
                L = loss(out, y)
                # backward pass
                opt.zero_grad()
                L.backward()
                opt.step()
                
            with t.no_grad():
                logits = self.probe(X_test).squeeze(-1)
                probs = F.sigmoid(logits)
                fpr, tpr, _ = metrics.roc_curve(Y_test.cpu().float(), probs.detach().cpu().float())
                auroc = metrics.auc(fpr, tpr)
                print(f"epoch {epoch} - test AUROC: {auroc:.3f}")

    def _fit_mean_difference(self) -> None:
        print(f"fitting {self.probe_type} probe")
        
        # compute means on raw activations
        honest_mean = self.honest.mean(dim=0)
        dishonest_mean = self.dishonest.mean(dim=0)
        
        # probe direction: dishonest - honest (so positive = more dishonest)
        probe_direction = dishonest_mean - honest_mean
        
        # normalize to unit vector
        probe_direction = probe_direction / probe_direction.norm()
        
        # set the probe weights
        with t.no_grad():
            self.probe.weight.data = probe_direction.unsqueeze(0)
        
        # evaluate on full dataset using raw activations
        X_full = t.cat((self.honest, self.dishonest), dim=0)
        Y_full = t.tensor(
            [0 for _ in range(len(self.honest))] + [1 for _ in range(len(self.dishonest))],
            dtype=t.float32
        )
        
        with t.no_grad():
            logits = self.probe(X_full).squeeze(-1)
            probs = F.sigmoid(logits)
            fpr, tpr, _ = metrics.roc_curve(Y_full.cpu().float(), probs.detach().cpu().float())
            auroc = metrics.auc(fpr, tpr)
            print(f"mean-difference probe AUROC: {auroc:.3f}")

    def _fit_ccs(self, nepoch: int = 100, lr: float = 1e-3, consistency_weight: float = 1.0, confidence_weight: float = 1.0, lambda_l2: float = 1.0) -> None:
        print(f"fitting {self.probe_type} probe")
        
        # use raw activations for CCS
        honest = self.honest
        dishonest = self.dishonest
        
        # ensure we have the same number of tokens for contrast pairs
        assert len(honest) == len(dishonest), "honest and dishonest must have same length for CCS"
        
        # initialize probe randomly
        nn.init.normal_(self.probe.weight, std=0.01)
        
        # optimizer
        opt = t.optim.AdamW(self.probe.parameters(), lr=lr)

        # evaluate on full dataset using raw activations
        X_full = t.cat((honest, dishonest), dim=0)
        Y_full = t.tensor(
            [0 for _ in range(len(honest))] + [1 for _ in range(len(dishonest))],
            dtype=t.float32
        )
        
        for epoch in range(nepoch):
            opt.zero_grad()
            
            # get probe outputs (logits)
            honest_logits = self.probe(honest).squeeze(-1)
            dishonest_logits = self.probe(dishonest).squeeze(-1)
            
            # convert to probabilities
            honest_probs = t.sigmoid(honest_logits)
            dishonest_probs = t.sigmoid(dishonest_logits)
            
            # Burns' CCS loss formulation
            # consistency loss: p0 should equal 1-p1 (opposite predictions)
            consistency_loss = ((honest_probs - (1 - dishonest_probs)) ** 2).mean()
            
            # informative loss: at least one probability should be small (prevents 0.5 degeneracy)
            informative_loss = (t.min(honest_probs, dishonest_probs) ** 2).mean()
            
            # total CCS loss
            ccs_loss = consistency_loss + informative_loss
            
            # add L2 regularization
            l2_loss = lambda_l2 * self.probe.weight.pow(2).sum()
            total_loss = ccs_loss + l2_loss
            
            # backward pass
            total_loss.backward()
            opt.step()
            
            if epoch % 20 == 0:
                print(f"epoch {epoch} - consistency: {consistency_loss.item():.4f}, informative: {informative_loss.item():.4f}, l2: {l2_loss.item():.4f}")
                with t.no_grad():
                    logits = self.probe(X_full).squeeze(-1)
                    probs = F.sigmoid(logits)
                    fpr, tpr, _ = metrics.roc_curve(Y_full.cpu().float(), probs.detach().cpu().float())
                    auroc = metrics.auc(fpr, tpr)
                    print(f"epoch {epoch} - AUROC: {auroc:.3f}")
        
        # ensure correct orientation: dishonest should have higher scores than honest
        with t.no_grad():
            honest_scores = self.probe(honest).squeeze(-1).mean()
            dishonest_scores = self.probe(dishonest).squeeze(-1).mean()
            
            if honest_scores > dishonest_scores:
                # flip the probe direction
                self.probe.weight.data = -self.probe.weight.data
                print("flipped probe direction to ensure dishonest > honest")
        
        with t.no_grad():
            logits = self.probe(X_full).squeeze(-1)
            probs = F.sigmoid(logits)
            fpr, tpr, _ = metrics.roc_curve(Y_full.cpu().float(), probs.detach().cpu().float())
            auroc = metrics.auc(fpr, tpr)
            print(f"CCS probe AUROC: {auroc:.3f}")
            
            # also report final losses
            honest_probs_eval = t.sigmoid(self.probe(honest).squeeze(-1))
            dishonest_probs_eval = t.sigmoid(self.probe(dishonest).squeeze(-1))
            consistency_eval = ((honest_probs_eval - (1 - dishonest_probs_eval)) ** 2).mean()
            informative_eval = (t.min(honest_probs_eval, dishonest_probs_eval) ** 2).mean()
            print(f"Final consistency loss: {consistency_eval.item():.4f}")
            print(f"Final informative loss: {informative_eval.item():.4f}")

    def _fit_crc_tpc(self) -> None:
        print(f"fitting {self.probe_type} probe")
        
        # ensure we have the same number of tokens for contrast pairs
        assert len(self.honest) == len(self.dishonest), "honest and dishonest must have same length for CRC-TPC"
        
        # move data to GPU for faster computation
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        honest_gpu = self.honest.to(device)
        dishonest_gpu = self.dishonest.to(device)
        
        # compute differences: dishonest - honest for each contrast pair
        differences = dishonest_gpu - honest_gpu  # shape: [n_samples, d_model]
        
        # center the differences
        differences_centered = differences - differences.mean(dim=0)
        
        # use SVD instead of eigendecomposition - much more robust
        U, S, Vt = t.linalg.svd(differences_centered, full_matrices=False)
        
        # first principal component is the first column of V (or first row of Vt)
        top_pc = Vt[0, :]  # shape: [d_model]
        
        # ensure direction points from honest to dishonest
        # (check if it aligns with mean difference)
        mean_diff = differences.mean(dim=0)
        if t.dot(top_pc, mean_diff) < 0:
            top_pc = -top_pc
        
        # normalize to unit vector (should already be normalized from SVD, but just to be safe)
        top_pc = top_pc / top_pc.norm()
        
        # move back to CPU and set the probe weights
        with t.no_grad():
            self.probe.weight.data = top_pc.cpu().unsqueeze(0)
        
        # evaluate on full dataset using raw activations (CPU)
        X_full = t.cat((self.honest, self.dishonest), dim=0)
        Y_full = t.tensor(
            [0 for _ in range(len(self.honest))] + [1 for _ in range(len(self.dishonest))],
            dtype=t.float32
        )
        
        with t.no_grad():
            logits = self.probe(X_full).squeeze(-1)
            probs = F.sigmoid(logits)
            fpr, tpr, _ = metrics.roc_curve(Y_full.cpu().float(), probs.detach().cpu().float())
            auroc = metrics.auc(fpr, tpr)
            print(f"CRC-TPC probe AUROC: {auroc:.3f}")
            
            # report explained variance (move S back to CPU for this)
            explained_var = S[0]**2 / (S**2).sum()
            print(f"Top PC explains {explained_var.cpu().item():.3f} of variance in differences")


if __name__ == "__main__":
    models = os.listdir(f"{CACHE_PATH}/activations/")
    for model_name in models:
        print(f"fitting probes for {model_name}")
        for probe_type in ["supervised", "diff-in-means", "ccs", "crc-tpc"]:
            output_path = f"{CACHE_PATH}/probes/{model_name}"
            if os.path.exists(f"{output_path}/{probe_type}.pt"):
                print(f"probe {probe_type} for {model_name} already exists")
                continue
            print("="*100)
            probe = Probe(model_name, probe_type)
            probe.fit()
            os.makedirs(output_path, exist_ok=True)
            t.save(probe.probe.weight.data.to(t.bfloat16), f"{output_path}/{probe_type}.pt")
            if probe_type == "supervised":
                t.save(probe.mu.to(t.bfloat16), f"{output_path}/mu.pt")
                t.save(probe.sigma.to(t.bfloat16), f"{output_path}/sigma.pt")
        print()
        print()