import os
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
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
        honest = t.load(f"{path}/honest.pt", weights_only=False)
        dishonest = t.load(f"{path}/dishonest.pt", weights_only=False)

        # === Z-SCORE STATS ===
        mu = t.cat((honest, dishonest), dim=0).mean(dim=0)
        sigma = t.cat((honest, dishonest), dim=0).std(dim=0)
        # apply z-score
        self.honest = (honest - mu) / sigma
        self.dishonest = (dishonest - mu) / sigma
        self.mu = mu
        self.sigma = sigma

        # init probe
        d_model = honest.shape[1]
        self.probe = nn.Linear(d_model, 1, dtype=t.bfloat16, bias=False)

        self.probe_type = probe_type
        self.split = split

    def forward(self, x) -> t.Tensor:
        return self.probe(x)

    def fit(self, **kwargs) -> None:
        if self.probe_type == "supervised":
            self._fit_supervised(**kwargs)
        else:
            raise ValueError(f"probe type {self.probe_type} not supported")
    
    def _fit_supervised(self, batch_size: int = 64, nepoch: int = 5) -> None:
        print(f"fitting {self.probe_type} probe")
        # prep training data
        X = t.cat((self.honest, self.dishonest), dim=0)
        Y = t.tensor(
            [0 for _ in range(len(self.honest))] +
            [1 for _ in range(len(self.dishonest))],
            dtype=t.bfloat16
        )
        # random shuffle
        perm = t.randperm(len(X))
        X, Y = X[perm], Y[perm]
        # train/test split
        split = int(self.split * len(X))
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]
        # batch data
        nbatch = len(X_train) // batch_size
        # prepare probe
        opt = t.optim.AdamW(self.probe.parameters(), lr=1e-3, weight_decay=1e-4)
        loss = nn.BCEWithLogitsLoss()
        # train
        for epoch in range(nepoch):
            # reshuffle training data
            perm = t.randperm(len(X_train))
            X_train, Y_train = X_train[perm], Y_train[perm]
            for b in trange(nbatch, desc=f"epoch {epoch}"):
                x, y = X_train[b*batch_size:(b+1)*batch_size], Y_train[b*batch_size:(b+1)*batch_size]
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
            print(f"test AUROC: {auroc:.3f}")


if __name__ == "__main__":
    models = os.listdir(f"{CACHE_PATH}/activations/")
    for model_name in models:
        print(f"fitting probes for {model_name}")
        for probe_type in ["supervised"]:
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
            t.save(probe.mu, f"{output_path}/mu.pt")
            t.save(probe.sigma, f"{output_path}/sigma.pt")
        print()
        print()