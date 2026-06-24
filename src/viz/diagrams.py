"""Programmatic architecture / control-loop diagram for the paper and website."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from ..utils import FIGURES_DIR, ensure_dir


def _box(ax, x, y, w, h, text, fc, ec="#333", fs=9.5, tc="#111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def _arrow(ax, x0, y0, x1, y1, color="#333", style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=13, lw=1.4, color=color))


def architecture(outname="architecture.png"):
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    _box(ax, 0.2, 2.1, 1.5, 0.9, "reference\nmodel", "#eaf2fb")
    _box(ax, 2.1, 2.1, 2.1, 0.9, "computed-torque\nPD baseline", "#e9e9ec")
    _box(ax, 4.7, 2.1, 2.3, 0.9, "stability-constrained\nneural adaptive term", "#dcebf7", ec="#1f6fb2")
    _box(ax, 7.5, 2.1, 2.0, 0.9, "plant\n(MuJoCo arm)", "#e9f6ec", ec="#2e7d44")

    # signal flow
    _arrow(ax, 1.7, 2.55, 2.1, 2.55)
    _arrow(ax, 4.2, 2.55, 4.7, 2.55)
    _arrow(ax, 7.0, 2.55, 7.5, 2.55)
    ax.text(7.25, 2.75, r"$\tau$", fontsize=10)

    # feedback
    _arrow(ax, 8.5, 2.1, 8.5, 1.2, color="#555")
    _arrow(ax, 8.5, 1.2, 1.0, 1.2, color="#555")
    _arrow(ax, 1.0, 1.2, 1.0, 2.1, color="#555")
    ax.text(4.6, 1.0, r"state $q,\dot q$  (error $e=q-x_\mathrm{ref}$)", fontsize=8.5, color="#555")

    # drift injection
    _box(ax, 7.6, 3.7, 1.8, 0.7, "drift injection\nmass / friction / damping", "#fdecea", ec="#c0392b", fs=8)
    _arrow(ax, 8.5, 3.7, 8.5, 3.0, color="#c0392b")

    # Lyapunov update + certificate
    _box(ax, 4.7, 3.7, 2.3, 0.7, r"Lyapunov update  $\dot\Theta=\Gamma\,\phi(x)(e^\top PB)-\sigma\Gamma\Theta$",
         "#dcebf7", ec="#1f6fb2", fs=7.5)
    _arrow(ax, 5.85, 3.7, 5.85, 3.0, color="#1f6fb2")

    _box(ax, 4.7, 0.2, 2.3, 0.7, "neural Lyapunov certificate\n(independent confirmation)",
         "#f3eafc", ec="#7a4fb0", fs=7.5)
    _arrow(ax, 5.85, 1.05, 5.85, 0.9, color="#7a4fb0", style="-|>")

    ax.set_title("Stability-guaranteed neural adaptation under mid-task dynamics drift", fontsize=10.5)
    fig.tight_layout()
    out = ensure_dir(FIGURES_DIR) / outname
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(architecture())
