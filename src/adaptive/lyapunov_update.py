"""The Lyapunov-derived adaptive update law and its projection operator.

Candidate Lyapunov function:
    V = e_aug^T P e_aug + tr(Theta_tilde^T Gamma^{-1} Theta_tilde)

Differentiating along the closed loop and choosing

    Theta_dot = Gamma * phi(x) * (e_aug^T P B) - sigma * Gamma * Theta      (sigma-modification)

cancels the cross term in V_dot and leaves V_dot = -e_aug^T Q e_aug + (approx-error terms),
negative definite in e_aug outside a small residual set. The sigma-modification plus a
projection onto a bounded ball keep Theta bounded under nonzero approximation error, which is
what makes the claim uniformly-ultimately-bounded rather than asymptotic-to-zero.
"""
from __future__ import annotations
import numpy as np


def constrained_theta_dot(phi, e_aug, P, B, gamma, theta, sigma):
    """Lyapunov-derived rate of change of the adaptive weights Theta (shape m x n)."""
    w = B.T @ (P @ e_aug)          # (n,)  = e_aug^T P B
    return gamma * np.outer(phi, w) - sigma * gamma * theta


def project_columns(theta, theta_max):
    """Projection operator: clip each output column of Theta to norm <= theta_max."""
    out = theta.copy()
    norms = np.linalg.norm(out, axis=0)
    for j, nrm in enumerate(norms):
        if nrm > theta_max and nrm > 0:
            out[:, j] *= theta_max / nrm
    return out


def lyapunov_value(e_aug, P, theta, gamma):
    """V used for the violation count: error energy plus weight energy (Theta* = 0 reference)."""
    return float(e_aug @ P @ e_aug + np.sum(theta ** 2) / gamma)
