"""Contour figure for the learned neural Lyapunov certificate."""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..utils import FIGURES_DIR, ensure_dir


def plot_certificate(cert_json, outname="e1_certificate.png"):
    d = json.load(open(cert_json))
    g = np.array(d["grid_e"]); gd = np.array(d["grid_ed"])
    V = np.array(d["V_grid"]); dec = np.array(d["decrease_grid"])
    GX, GY = np.meshgrid(g, gd)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    cs = ax.contourf(GX, GY, V, levels=18, cmap="viridis", alpha=0.95)
    ax.contour(GX, GY, V, levels=8, colors="white", linewidths=0.4, alpha=0.5)
    # shade the region where the learned decrease condition holds
    ax.contourf(GX, GY, dec, levels=[0.5, 1.5], colors=["none"], hatches=["///"], alpha=0)
    ax.contour(GX, GY, dec, levels=[0.5], colors="#c0392b", linewidths=1.4)
    cb = fig.colorbar(cs, ax=ax); cb.set_label("$V_\\theta(x)$")
    ax.set_xlabel("joint-1 error  $e_1$ (rad)")
    ax.set_ylabel("joint-1 error rate  $\\dot{e}_1$ (rad/s)")
    frac = d["verify_fraction"] * 100
    ax.set_title(f"Neural Lyapunov certificate\n{frac:.1f}% of sampled states satisfy "
                 f"$V_\\theta(x_{{k+1}}) \\leq (1-\\alpha\\,dt)V_\\theta(x_k)$")
    ax.text(0.02, 0.02, "red contour: boundary of learned decrease region",
            transform=ax.transAxes, fontsize=7, color="#c0392b")
    fig.tight_layout()
    out = ensure_dir(FIGURES_DIR) / outname
    fig.savefig(out)
    plt.close(fig)
    return out
