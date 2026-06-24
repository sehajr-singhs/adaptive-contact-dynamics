"""A learned neural Lyapunov certificate for the closed-loop adaptive error dynamics.

This is independent confirmation, separate from the controller. We parameterize a candidate
Lyapunov function that is positive definite by construction,

    V_theta(x) = ||psi(x)||^2 + eps ||x||^2,   psi a small MLP,

so V_theta >= eps ||x||^2 > 0 away from the origin without any training. We then train it on real
transition pairs (x_k, x_{k+1}) collected from the constrained controller running on the drifted
plant, penalizing violations of the exponential decrease condition

    V_theta(x_{k+1}) <= (1 - alpha dt) V_theta(x_k).

Verification is sampling-based over the operating region, which is honest about what a learned
certificate buys: confidence over a bounded region, not a formal global proof.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class LyapunovNet(nn.Module):
    def __init__(self, dim, hidden=64, eps=0.05):
        super().__init__()
        self.eps = eps
        self.psi = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x):
        p = self.psi(x)
        return (p * p).sum(-1) + self.eps * (x * x).sum(-1)


def train_certificate(X0, X1, dt, alpha=0.2, hidden=64, eps=0.05,
                      epochs=400, lr=2e-3, seed=0):
    """Fit V_theta on transition pairs (X0 -> X1). Returns (model, history)."""
    torch.manual_seed(seed)
    dim = X0.shape[1]
    model = LyapunovNet(dim, hidden=hidden, eps=eps)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x0 = torch.tensor(X0, dtype=torch.float32)
    x1 = torch.tensor(X1, dtype=torch.float32)
    decay = 1.0 - alpha * dt
    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        v0 = model(x0)
        v1 = model(x1)
        # decrease violation, hinge; plus a mild margin so V is not driven to the eps floor
        viol = torch.relu(v1 - decay * v0)
        loss = viol.mean() + 1e-3 * torch.relu(0.02 - v0).mean()
        loss.backward()
        opt.step()
        hist.append(loss.item())
    return model, hist


def verify_certificate(model, X0, X1, dt, alpha=0.2):
    """Fraction of held-out transitions satisfying the decrease condition + worst violation."""
    decay = 1.0 - alpha * dt
    with torch.no_grad():
        v0 = model(torch.tensor(X0, dtype=torch.float32)).numpy()
        v1 = model(torch.tensor(X1, dtype=torch.float32)).numpy()
    ok = v1 <= decay * v0 + 1e-6
    worst = float(np.max(v1 - decay * v0))
    pos = float(np.mean(v0 > 0))
    return {
        "verify_fraction": float(np.mean(ok)),
        "worst_violation": worst,
        "positive_fraction": pos,
        "n_verify": int(len(X0)),
    }
