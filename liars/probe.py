import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        activation_path = f"{CACHE_PATH}/activations/{model_name}"
        activations = t.load(f"{activation_path}/activations.pt", weights_only=False)
        labels = t.load(f"{activation_path}/labels.pt", weights_only=False)
        labels = (~labels).to(t.bfloat16)
        assert len(labels) == len(activations)

        # train/test split now as we use standard activations for testing
        perm = t.randperm(len(activations))
        activations, labels = activations[perm], labels[perm]
        split = int(0.8 * len(activations))
        self.X_train, self.X_test = activations[:split], activations[split:]
        self.Y_train, self.Y_test = labels[:split], labels[split:]

        self.d_model = self.X_train.shape[-1]
        self.probe = nn.Linear(self.d_model, 1, dtype=t.bfloat16, bias=False)

    def forward(self, x) -> t.Tensor:
        return self.probe(x)

    def eval(self, plot: bool = False) -> None:
        with t.no_grad():
            logits = self.probe(self.X_test).squeeze(-1)
            probs = F.sigmoid(logits)
            fpr, tpr, _ = metrics.roc_curve(self.Y_test.cpu().float(), probs.detach().cpu().float())
            auroc = metrics.auc(fpr, tpr)
            print(f"test AUROC: {auroc:.3f}")
            if plot:
                mask = self.Y_test.bool().numpy()
                scores = self.probe(self.X_test).squeeze(-1).float().numpy()
                plt.hist(scores[mask], bins=25, label="truths", alpha=0.5, color="blue")
                plt.hist(scores[~mask], bins=25, label="lies", alpha=0.5, color="red")
                plt.title(f"probe scores")
                plt.legend()
                plt.show()

    def fit(self, **kwargs) -> None:
        if self.probe_type == "supervised":
            self._fit_supervised(**kwargs)
        elif self.probe_type == "diff-in-means":
            self._fit_diff_in_means()
        else:
            raise ValueError(f"probe type {self.probe_type} not supported")

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
        print(f"fitting probes for {model_name}")
        for probe_type in ["supervised", "diff-in-means"]:
            output_path = f"{CACHE_PATH}/probes/{model_name}"
            if os.path.exists(f"{output_path}/{probe_type}.pt"):
                print(f"probe {probe_type} for {model_name} already exists")
                continue
            print("="*100)
            probe = Probe(model_name, probe_type)
            probe.fit(nepoch=10)
            probe.eval()
            os.makedirs(output_path, exist_ok=True)
            t.save(probe.probe.weight.data, f"{output_path}/{probe_type}.pt")
        print()
        print()