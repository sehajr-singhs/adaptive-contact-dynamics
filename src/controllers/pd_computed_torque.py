"""Computed-torque PD baseline for the arm environments.

The torque law is shared by every controller; the adaptive ones simply add their u_ad into
the commanded acceleration. Computed against the NOMINAL model, so model drift becomes a
matched uncertainty:

    tau = M0(q) (x_ref_ddot - Kp e - Kd e_dot + u_ad) + b0(q, q_dot)

Under the nominal model and u_ad = 0 this gives e_ddot = -Kp e - Kd e_dot, i.e. the Hurwitz
reference dynamics. The fixed baseline is exactly this with u_ad = 0; it is what degrades when
the physics drift, and it is tuned before drift so the comparison is not a strawman.
"""
from __future__ import annotations
import numpy as np


def computed_torque(M0, b0, xdd_ref, e, ed, kp, kd, u_ad):
    # Error dynamics are e_ddot = -Kp e - Kd e_dot + (Delta - u_ad), so u_ad is SUBTRACTED:
    # the Lyapunov-derived law then drives u_ad -> Delta and cancels the matched uncertainty.
    a_des = xdd_ref - kp * e - kd * ed - u_ad
    return M0 @ a_des + b0


class FixedBaseline:
    """Non-adaptive controller: u_ad is identically zero, nothing updates online."""

    name = "fixed_baseline"

    def __init__(self, n_out: int):
        self.n_out = n_out

    def reset(self):
        pass

    def u_ad(self, state):
        return np.zeros(self.n_out)

    def update(self, state, e_aug):
        pass

    def V(self, e_aug):
        return float("nan")
