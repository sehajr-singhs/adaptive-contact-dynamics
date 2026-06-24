"""Paper-quality figures, all regenerated from saved results JSON (no re-simulation).

Visual language follows the reference site: calm, technical, minimal, mean +/- std bands, the
drift event marked, an honest per-environment comparison. One clean style, no chartjunk.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..utils import RESULTS_DIR, FIGURES_DIR, ensure_dir

CTRL_ORDER = ["fixed_baseline", "constrained_adaptive", "unconstrained_adaptive"]
CTRL_LABEL = {
    "fixed_baseline": "Fixed baseline",
    "constrained_adaptive": "Constrained adaptive (ours)",
    "unconstrained_adaptive": "Unconstrained adaptive (ablation)",
}
CTRL_COLOR = {
    "fixed_baseline": "#7a7a7a",
    "constrained_adaptive": "#1f6fb2",
    "unconstrained_adaptive": "#c0392b",
}

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
})


def _load_seed_series(env, controller, regime, key, seeds):
    series = []
    for s in seeds:
        p = RESULTS_DIR / env / controller / regime / f"seed{s}.json"
        if p.exists():
            series.append(np.asarray(json.load(open(p))[key], dtype=float))
    if not series:
        return None
    L = min(len(x) for x in series)
    return np.stack([x[:L] for x in series])


def _meta(env, controller, regime, seeds):
    for s in seeds:
        p = RESULTS_DIR / env / controller / regime / f"seed{s}.json"
        if p.exists():
            return json.load(open(p))
    return None


def plot_tracking_error(env, regime, seeds, title=None, outname=None):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    drift_step = None
    dt = 0.01
    # scale the y-axis to the stable controllers; an unstable ablation is annotated off-scale
    scale_peak = 0.0
    peaks = {}
    series = {}
    for c in CTRL_ORDER:
        arr = _load_seed_series(env, c, regime, "err_deg_series", seeds)
        if arr is None:
            continue
        series[c] = arr
        peaks[c] = float(np.nanmax(arr.mean(0)))
        if c in ("fixed_baseline", "constrained_adaptive"):
            scale_peak = max(scale_peak, peaks[c])
    ymax = max(1.0, scale_peak * 1.5)
    for c in CTRL_ORDER:
        if c not in series:
            continue
        arr = series[c]
        m = _meta(env, c, regime, seeds)
        drift_step, dt = m["drift_step"], m["dt"]
        t = np.arange(arr.shape[1]) * dt
        mu, sd = arr.mean(0), arr.std(0)
        ax.plot(t, mu, color=CTRL_COLOR[c], label=CTRL_LABEL[c], lw=1.8)
        # only draw the std band for controllers that stay on-scale, else it washes the plot out
        if peaks[c] <= ymax:
            ax.fill_between(t, mu - sd, mu + sd, color=CTRL_COLOR[c], alpha=0.15)
    if drift_step is not None:
        ax.axvline(drift_step * dt, color="k", ls="--", lw=1, alpha=0.6)
        ax.text(drift_step * dt, ymax * 0.5, " drift", fontsize=8, va="top")
    unit = (_meta(env, "fixed_baseline", regime, seeds) or {}).get("error_unit", "deg")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"tracking error ({unit}, RMS)")
    ax.set_title(title or f"{env} | {regime} drift")
    ax.set_ylim(0, ymax)
    # annotate any controller that diverges off the top of the axis
    unc = "unconstrained_adaptive"
    if unc in peaks and peaks[unc] > ymax:
        ax.text(0.97, 0.92, f"unconstrained ablation (matched gain)\ndiverges off-scale, "
                f"peak {peaks[unc]:.0f} {unit}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                color=CTRL_COLOR[unc])
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = ensure_dir(FIGURES_DIR) / (outname or f"{env}_{regime}_tracking.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_energy(env, regime, seeds, outname=None):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    drift_step, dt = None, 0.01
    for c in CTRL_ORDER:
        arr = _load_seed_series(env, c, regime, "energy_series", seeds)
        if arr is None:
            continue
        m = _meta(env, c, regime, seeds)
        drift_step, dt = m["drift_step"], m["dt"]
        bound = m.get("energy_bound", None)
        t = np.arange(arr.shape[1]) * dt
        mu = np.clip(arr.mean(0), 1e-4, None)
        ax.plot(t, mu, color=CTRL_COLOR[c], label=CTRL_LABEL[c], lw=1.8)
    if drift_step is not None:
        ax.axvline(drift_step * dt, color="k", ls="--", lw=1, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("tracking-error energy  $e^\\top P e$")
    ax.set_title(f"{env} | {regime} drift | error energy (log scale)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = ensure_dir(FIGURES_DIR) / (outname or f"{env}_{regime}_energy.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_recovery_bar(env, regimes, seeds, outname=None):
    """Post-drift steady-state error per controller, mean +/- std, grouped by regime."""
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    width = 0.25
    x = np.arange(len(regimes))
    for i, c in enumerate(CTRL_ORDER):
        means, stds = [], []
        for rg in regimes:
            vals = []
            for s in seeds:
                p = RESULTS_DIR / env / c / rg / f"seed{s}.json"
                if p.exists():
                    vals.append(json.load(open(p))["ss_error_post_deg"])
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3,
               color=CTRL_COLOR[c], label=CTRL_LABEL[c])
    unit = (_meta(env, "fixed_baseline", regimes[0], seeds) or {}).get("error_unit", "deg")
    ax.set_xticks(x)
    ax.set_xticklabels([r + " drift" for r in regimes])
    ax.set_ylabel(f"post-drift error ({unit})")
    ax.set_title(f"{env} | post-drift steady-state tracking error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = ensure_dir(FIGURES_DIR) / (outname or f"{env}_recovery_bar.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_gain_sweep(sweep_json, outname="e1_gain_sweep.png"):
    """The stability-ceiling figure: post-drift error vs adaptation gain for both methods."""
    d = json.load(open(sweep_json))
    g = np.array(d["gammas"])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for c in ["constrained_adaptive", "unconstrained_adaptive"]:
        mu = np.array(d[c]["mean"])
        sd = np.array(d[c]["std"])
        ax.plot(g, mu, "-o", color=CTRL_COLOR[c], label=CTRL_LABEL[c], lw=1.8, ms=4)
        ax.fill_between(g, mu - sd, mu + sd, color=CTRL_COLOR[c], alpha=0.15)
    if "ceiling" in d and d["ceiling"]:
        ax.axvline(d["ceiling"], color="#c0392b", ls=":", lw=1.2)
        ax.text(d["ceiling"], ax.get_ylim()[1] * 0.9,
                " unconstrained\n stability ceiling", fontsize=7, color="#c0392b")
    ax.set_xscale("log")
    ax.set_xlabel("adaptation gain $\\gamma$")
    ax.set_ylabel("post-drift error (deg)")
    ax.set_title("Stability ceiling: only the constrained law stays stable as $\\gamma$ grows")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = ensure_dir(FIGURES_DIR) / outname
    fig.savefig(out)
    plt.close(fig)
    return out
