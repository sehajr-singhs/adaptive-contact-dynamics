"""MRAC backbone: reference model, error-system matrices, and the Lyapunov P.

For an n-joint arm we stack the per-joint tracking error e = q - x_ref and its rate into
e_aug = [e, e_dot] in R^{2n}. With the computed-torque PD feedback (gains Kp, Kd) the
nominal closed loop is

    e_dot_aug = A_ref e_aug + B (Delta(x) - u_ad)

    A_ref = [[0, I], [-Kp, -Kd]]   (Hurwitz by construction)
    B     = [[0], [I]]

Delta is the matched uncertainty (the part of the true dynamics the nominal model misses,
including the drift), and u_ad is the adaptive term that cancels it. A_ref Hurwitz gives a
P = P^T > 0 solving A_ref^T P + P A_ref = -Q, computed once here.
"""
from __future__ import annotations
import numpy as np
import scipy.linalg


class ReferenceModel:
    """Second-order reference per joint: well-damped, fixed bandwidth.

    x_ref_ddot = -wn^2 (x_ref - x_cmd) - 2 zeta wn x_ref_dot
    """

    def __init__(self, n: int, wn: float, zeta: float, dt: float):
        self.n = n
        self.wn = wn
        self.zeta = zeta
        self.dt = dt
        self.x = np.zeros(n)
        self.xd = np.zeros(n)
        self.xdd = np.zeros(n)

    def reset(self, x0):
        self.x = np.array(x0, dtype=float)
        self.xd = np.zeros(self.n)
        self.xdd = np.zeros(self.n)

    def step(self, x_cmd):
        self.xdd = -(self.wn ** 2) * (self.x - x_cmd) - 2 * self.zeta * self.wn * self.xd
        self.xd = self.xd + self.dt * self.xdd
        self.x = self.x + self.dt * self.xd
        return self.x.copy(), self.xd.copy(), self.xdd.copy()


def build_error_system(n: int, kp: float, kd: float, q_scale: float = 1.0):
    """Return A_ref, B, P, Q for an n-joint error system with scalar PD gains."""
    I = np.eye(n)
    Z = np.zeros((n, n))
    A_ref = np.block([[Z, I], [-kp * I, -kd * I]])
    B = np.vstack([Z, I])  # (2n, n)
    Q = q_scale * np.eye(2 * n)
    # A_ref^T P + P A_ref = -Q  (continuous Lyapunov equation)
    P = scipy.linalg.solve_continuous_lyapunov(A_ref.T, -Q)
    P = 0.5 * (P + P.T)
    return A_ref, B, P, Q
