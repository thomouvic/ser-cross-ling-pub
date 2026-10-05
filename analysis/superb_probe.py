"""SUPERB-style weighted-sum probe.

Linear classifier on top of a learnable weighted sum across the encoder's
hidden states. Trained jointly via cross-entropy + Adam.

Usage:
    from superb_probe import SuperbProbe
    probe = SuperbProbe(n_layers=13, dim=768, n_classes=4)
    probe.fit(X_train, y_train)            # X_train: (N, L, D)
    pred = probe.predict(X_test)           # X_test: (M, L, D) -> (M,)
    weights = probe.layer_weights()         # softmaxed alphas, (L,)

The weighted sum is `Σᵢ softmax(w)ᵢ · X[:, i, :]`. Encoder is frozen — these
embeddings are precomputed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Net(nn.Module):
    def __init__(self, n_layers: int, dim: int, n_classes: int, hidden: int = 0):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))
        if hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
            )
        else:
            self.head = nn.Linear(dim, n_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, L, D)
        alphas = self.w.softmax(0)            # (L,)
        combined = (X * alphas[None, :, None]).sum(dim=1)  # (B, D)
        return self.head(combined)


class SuperbProbe:
    """SUPERB-style probe: learnable layer weights + linear head."""

    def __init__(self, n_layers: int, dim: int, n_classes: int,
                 hidden: int = 0, lr: float = 1e-3, weight_decay: float = 1e-4,
                 epochs: int = 200, batch_size: int = 256, verbose: bool = False,
                 device: str = "cpu", seed: int = 20260505,
                 standardize: bool = True):
        self.n_layers = n_layers
        self.dim = dim
        self.n_classes = n_classes
        self.hidden = hidden
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = device
        self.seed = seed
        self.standardize = standardize
        self.net: _Net | None = None
        self.scaler_mean: torch.Tensor | None = None
        self.scaler_std: torch.Tensor | None = None

    def _standardize_fit(self, X: torch.Tensor) -> torch.Tensor:
        if not self.standardize:
            # No-op scaler (identity) so _standardize_apply also passes through.
            self.scaler_mean = torch.zeros_like(X[:1])
            self.scaler_std = torch.ones_like(X[:1])
            return X
        # Standardize per-(layer, dim) over the training set
        self.scaler_mean = X.mean(dim=0, keepdim=True)
        self.scaler_std = X.std(dim=0, keepdim=True).clamp(min=1e-6)
        return (X - self.scaler_mean) / self.scaler_std

    def _standardize_apply(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.scaler_mean) / self.scaler_std

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SuperbProbe":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        Xt = torch.from_numpy(X).float().to(self.device)
        yt = torch.from_numpy(y).long().to(self.device)
        Xt = self._standardize_fit(Xt)
        self.net = _Net(self.n_layers, self.dim, self.n_classes, self.hidden).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        N = Xt.shape[0]
        for epoch in range(self.epochs):
            idx = torch.randperm(N, device=self.device)
            losses = []
            for start in range(0, N, self.batch_size):
                batch = idx[start:start + self.batch_size]
                xb, yb = Xt[batch], yt[batch]
                logits = self.net(xb)
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.verbose and (epoch % 25 == 0 or epoch == self.epochs - 1):
                with torch.no_grad():
                    pred = self.net(Xt).argmax(dim=-1)
                    acc = (pred == yt).float().mean().item()
                print(f"  epoch {epoch:3d}  train loss {np.mean(losses):.4f}  acc {acc:.3f}")
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        Xt = torch.from_numpy(X).float().to(self.device)
        Xt = self._standardize_apply(Xt)
        logits = self.net(Xt)
        return logits.argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xt = torch.from_numpy(X).float().to(self.device)
        Xt = self._standardize_apply(Xt)
        return torch.softmax(self.net(Xt), dim=-1).cpu().numpy()

    @torch.no_grad()
    def layer_weights(self) -> np.ndarray:
        return self.net.w.softmax(0).cpu().numpy()

    @torch.no_grad()
    def weighted_sum(self, X: np.ndarray) -> np.ndarray:
        """Return the weighted-sum representation (B, D) for use in RSA etc."""
        Xt = torch.from_numpy(X).float().to(self.device)
        Xt = self._standardize_apply(Xt)
        alphas = self.net.w.softmax(0)
        return (Xt * alphas[None, :, None]).sum(dim=1).cpu().numpy()
