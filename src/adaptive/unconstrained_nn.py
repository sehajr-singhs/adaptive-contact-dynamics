"""The unconstrained ablation: same feature map and same u_ad = Theta^T phi(x), but Theta is
updated by a plain gradient step on the instantaneous tracking-error loss with no P weighting,
no sigma-modification, and no projection. This is the single most important comparison in the
project, because it isolates whether the stability machinery is load-bearing or decorative.

We log its Lyapunov value with the *same* V used for the constrained design, so a violation
count is directly comparable. V is not guaranteed to decrease here, which is exactly the point.
"""
from __future__ import annotations
import numpy as np

from .stable_nn import FeatureMap
from .lyapunov_update import lyapunov_value


class UnconstrainedAdaptive:
    name = "unconstrained_adaptive"

    def __init__(self, feature: FeatureMap, n_out: int, P, B, dt, gamma=8.0):
        # P is held only so V() can be evaluated on the same Lyapunov function; the update
        # law below never uses it.
        self.feature = feature
        self.P, self.B, self.dt, self.gamma = P, B, dt, gamma
        self.theta = np.zeros((feature.out_dim, n_out))
        self._last_phi = None

    def reset(self):
        self.theta[:] = 0.0

    def u_ad(self, state):
        phi = self.feature(state)
        self._last_phi = phi
        return self.theta.T @ phi

    def update(self, state, e_aug):
        phi = self._last_phi if self._last_phi is not None else self.feature(state)
        # Plain gradient on 0.5||e_aug||^2: drive u_ad to reduce the raw tracking error.
        # No P (not Lyapunov-derived), no sigma leak, no projection.
        w = self.B.T @ e_aug
        self.theta = self.theta + self.dt * self.gamma * np.outer(phi, w)

    def V(self, e_aug):
        return lyapunov_value(e_aug, self.P, self.theta, self.gamma)
