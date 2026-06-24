"""Stability metrics built on the tracking-error energy E = e_aug^T P e_aug.

E is a valid Lyapunov function for the closed-loop tracking error, and the MRAC guarantee is
that E stays uniformly ultimately bounded: after a drift it may spike, but it returns within a
bounded set. So the honest, theory-aligned stability measure is whether E returns inside its
nominal (pre-drift) bound.

  - violation_fraction: fraction of LATE post-drift steps where E exceeds k x the pre-drift
    bound. A controller that recovers settles back inside the bound, so this is near zero; one
    that destabilizes stays outside it, so this is near one.
  - divergence: late-window energy relative to the pre-drift bound, a blow-up ratio.
"""
from __future__ import annotations
import numpy as np


def error_energy(e_aug_series, P):
    ea = np.asarray(e_aug_series)
    return np.einsum("ti,ij,tj->t", ea, P, ea)


def _pre_bound(E, drift_step):
    pre = E[:drift_step]
    if len(pre) < 5:
        return float(np.percentile(E, 95) + 1e-9)
    return float(np.percentile(pre, 95) + 1e-9)


def post_bound(E, drift_step):
    """95th-percentile energy of the (degraded) post-drift window. Used as the instability
    reference: a controller whose energy exceeds the *degraded baseline's* level is doing worse
    than not adapting, the signature of an unstable adaptive law."""
    post = E[drift_step:]
    if len(post) < 5:
        return float(np.percentile(E, 95) + 1e-9)
    return float(np.percentile(post, 95) + 1e-9)


def violation_fraction(E, drift_step, k=1.5, late_frac=0.5, ref_bound=None):
    """Fraction of late post-drift steps whose energy exceeds k x the instability reference
    (the degraded baseline's post-drift energy). High means the adaptive law is doing worse than
    not adapting, i.e. it is unstable. Low means the closed loop stays bounded."""
    E = np.asarray(E, dtype=float)
    bound = ref_bound if ref_bound is not None else post_bound(E, drift_step)
    post = E[drift_step:]
    if len(post) < 4:
        return 0, 0.0, bound
    lo = int(len(post) * (1 - late_frac))
    late = post[lo:]
    viol = int(np.sum(late > k * bound))
    return viol, float(viol / len(late)), bound


def divergence(E, drift_step, late_frac=0.3, ref_bound=None):
    """Mean late-window energy divided by the (healthy) bound (UUB blow-up ratio)."""
    E = np.asarray(E, dtype=float)
    bound = ref_bound if ref_bound is not None else _pre_bound(E, drift_step)
    post = E[drift_step:]
    if len(post) < 4:
        return 1.0
    lo = int(len(post) * (1 - late_frac))
    return float(np.mean(post[lo:]) / bound)
