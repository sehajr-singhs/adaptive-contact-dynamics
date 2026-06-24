"""Recovery and tracking-error metrics computed from a logged rollout.

The headline recovery number is the settling time of the tracking-error energy E = e_aug^T P e_aug
back inside its uniformly-ultimately-bounded set, because that is exactly what the MRAC guarantee
promises and it is robust to the fact that tracking error on a sinusoidal command oscillates. A
smoothed absolute-degree recovery time is kept as a secondary, more intuitive number.
"""
from __future__ import annotations
import numpy as np


def tracking_error(e_pos, scale=1.0):
    """RMS tracking error per timestep times a display scale (deg for arms, cm for push)."""
    e = np.asarray(e_pos)
    return np.sqrt(np.mean(e ** 2, axis=1)) * scale


def tracking_error_deg(e_pos):
    """RMS joint tracking error per timestep, in degrees. e_pos: (T, n) radians."""
    return tracking_error(e_pos, scale=180.0 / np.pi)


def smooth(x, w=30):
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def _settle_from(mask_ok, drift_step, dt):
    """First step after drift from which mask_ok holds to the end of the episode."""
    T = len(mask_ok)
    for k in range(drift_step, T):
        if np.all(mask_ok[k:]):
            return k - drift_step, (k - drift_step) * dt
    return None, None


def recovery_time_energy(E, drift_step, dt, bound, k=2.0):
    """Settling time: when E falls back below k x the pre-drift bound and stays there."""
    E = np.asarray(E, dtype=float)
    return _settle_from(E < k * bound, drift_step, dt)


def recovery_time(err_deg, drift_step, dt, threshold_deg, smooth_w=30):
    """Smoothed absolute-error recovery: when smoothed error returns below threshold to the end."""
    err = smooth(err_deg, smooth_w)
    return _settle_from(err < threshold_deg, drift_step, dt)


def steady_state_error(err_deg, lo, hi):
    return float(np.mean(np.asarray(err_deg)[lo:hi]))


def control_effort(taus, dt):
    tau = np.asarray(taus)
    return float(np.sum(tau ** 2) * dt)
