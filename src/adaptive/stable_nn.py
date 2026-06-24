"""The bounded neural feature map phi(x).

This is a radial-basis neural network, the canonical feature map for Lyapunov-stable neural
adaptive control (Sanner and Slotine, 1992). The inner layer is fixed: Gaussian basis units on a
regular grid over the operating region. The outer weights Theta adapt online under the Lyapunov
law. RBF outputs live in [0, 1], so ||phi(x)|| is bounded by construction, which is exactly the
boundedness assumption the uniformly-ultimately-bounded argument needs, and unlike a random tanh
projection the grid centers are deterministic so the guarantee does not hinge on a lucky draw.
The network supplies representation; the update law supplies the guarantee.

The `seed` argument is accepted for API symmetry but only jitters the grid by a sub-cell amount,
so different seeds are genuinely different feature maps yet all of them tile the region. This lets
the multi-seed study show the result is robust to the feature map rather than tuned to one.
"""
from __future__ import annotations
import itertools
import numpy as np


class FeatureMap:
    """Fixed bounded radial-basis feature map. Input = plant state, output in [0, 1]^m."""

    def __init__(self, in_dim: int, width: int = 64, layers: int = 2,
                 seed: int = 0, per_dim: int = 3, region=None, m_extra_bias: bool = True):
        # `width`/`layers` are kept in the signature for config compatibility; the RBF grid
        # resolution is set by `per_dim` (per_dim**in_dim basis units).
        self.in_dim = in_dim
        if region is None:
            # default operating region: joints ~[-1.5,1.5] rad, rates ~[-3,3] rad/s
            half = np.array([1.5] * (in_dim // 2) + [3.0] * (in_dim - in_dim // 2))
        else:
            half = np.asarray(region, dtype=float)
        self.scale = half
        rng = np.random.default_rng(seed)
        axis = np.linspace(-1.0, 1.0, per_dim)
        grid = np.array(list(itertools.product(axis, repeat=in_dim)))  # (per_dim^in_dim, in_dim)
        spacing = (axis[1] - axis[0]) if per_dim > 1 else 2.0
        jitter = rng.uniform(-0.25, 0.25, size=grid.shape) * spacing
        self.centers = grid + jitter
        self.gamma_rbf = 1.0 / (2.0 * (0.9 * spacing) ** 2)
        self.m_extra_bias = m_extra_bias
        self.out_dim = self.centers.shape[0] + (1 if m_extra_bias else 0)

    def __call__(self, x):
        xn = np.asarray(x, dtype=float) / self.scale
        d2 = np.sum((self.centers - xn) ** 2, axis=1)
        phi = np.exp(-self.gamma_rbf * d2)
        if self.m_extra_bias:
            phi = np.concatenate([phi, [1.0]])
        return phi


from .lyapunov_update import (
    constrained_theta_dot,
    project_columns,
    lyapunov_value,
)


class ConstrainedAdaptive:
    """Stability-constrained neural-adaptive term: u_ad = Theta^T phi(x).

    Theta evolves under the Lyapunov-derived law with sigma-modification and a column-norm
    projection, so the weights stay bounded and the closed loop is uniformly ultimately bounded
    under the matched-uncertainty and bounded-feature assumptions.
    """

    name = "constrained_adaptive"

    def __init__(self, feature: FeatureMap, n_out: int, P, B, dt,
                 gamma=8.0, sigma=0.02, theta_max=25.0):
        self.feature = feature
        self.P, self.B, self.dt = P, B, dt
        self.gamma, self.sigma, self.theta_max = gamma, sigma, theta_max
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
        td = constrained_theta_dot(phi, e_aug, self.P, self.B,
                                   self.gamma, self.theta, self.sigma)
        self.theta = self.theta + self.dt * td
        self.theta = project_columns(self.theta, self.theta_max)

    def V(self, e_aug):
        return lyapunov_value(e_aug, self.P, self.theta, self.gamma)
