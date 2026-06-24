"""Cross-environment master table + global summary figure, from saved results JSON."""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import RESULTS_DIR, FIGURES_DIR, ensure_dir, load_config

ENVS = [
    ("e1_planar_arm", "E1 planar arm"),
    ("e2_reacher", "E2 reacher"),
    ("e3_push", "E3 push"),
    ("e4_cube", "E4 OGBench cube"),
]
CTRLS = [
    ("fixed_baseline", "Fixed baseline"),
    ("constrained_adaptive", "Constrained (ours)"),
    ("unconstrained_adaptive", "Unconstrained (ablation)"),
]
COLOR = {"fixed_baseline": "#7a7a7a", "constrained_adaptive": "#1f6fb2",
         "unconstrained_adaptive": "#c0392b"}


def _agg(env, ctrl, regime, seeds):
    rows = []
    for s in seeds:
        p = RESULTS_DIR / env / ctrl / regime / f"seed{s}.json"
        if p.exists():
            rows.append(json.load(open(p)))
    if not rows:
        return None
    post = [r["ss_error_post_deg"] for r in rows]
    viol = [r["lyap_violation_frac"] for r in rows]
    rec = [r["recovery_seconds"] for r in rows]
    nrec = sum(1 for r in rec if r is not None)
    rec_secs = [r for r in rec if r is not None]
    return {
        "post_mean": float(np.mean(post)), "post_std": float(np.std(post)),
        "viol": float(np.mean(viol)), "nrec": nrec, "n": len(rows),
        "rec_mean": float(np.mean(rec_secs)) if rec_secs else None,
        "unit": rows[0].get("error_unit", "deg"),
    }


def build_table():
    lines = []
    lines.append("| Environment | Controller | Post-drift error (step) | Recovered | "
                 "Instability (viol.) |")
    lines.append("|---|---|---|---|---|")
    data = {}
    for env, elabel in ENVS:
        if not (RESULTS_DIR / env).exists():
            continue
        seeds = load_config(env)["run"]["seeds"]
        for ctrl, clabel in CTRLS:
            a = _agg(env, ctrl, "step", seeds)
            if a is None:
                continue
            data[(env, ctrl)] = a
            bold = "**" if ctrl == "constrained_adaptive" else ""
            lines.append(
                f"| {elabel} | {clabel} | {bold}{a['post_mean']:.2f} ± {a['post_std']:.2f} "
                f"{a['unit']}{bold} | {a['nrec']}/{a['n']} | {a['viol']:.2f} |")
    table = "\n".join(lines)
    out = ensure_dir(RESULTS_DIR) / "master_table.md"
    out.write_text(table, encoding="utf-8")
    json.dump({f"{e}|{c}": v for (e, c), v in data.items()},
              open(RESULTS_DIR / "master_table.json", "w"), indent=2)
    return table, data


def plot_global_summary(data, outname="global_summary.png"):
    """Post-drift error as a ratio to the fixed baseline, per environment (unitless, comparable)."""
    envs = [e for e, _ in ENVS if (e, "fixed_baseline") in data]
    labels = [lbl for e, lbl in ENVS if e in envs]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    width = 0.36
    x = np.arange(len(envs))
    for i, ctrl in enumerate(["constrained_adaptive", "unconstrained_adaptive"]):
        ratios = []
        for e in envs:
            base = data[(e, "fixed_baseline")]["post_mean"]
            val = data[(e, ctrl)]["post_mean"]
            ratios.append(val / base if base > 0 else np.nan)
        ax.bar(x + (i - 0.5) * width, ratios, width, color=COLOR[ctrl],
               label=CTRLS[1 + i][1])
    ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.7)
    ax.text(len(envs) - 0.5, 1.04, "fixed baseline", fontsize=7, ha="right")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("post-drift error / baseline\n(log; <1 is better than baseline)")
    ax.set_title("Adaptation vs the fixed baseline across environments (step drift)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = ensure_dir(FIGURES_DIR) / outname
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    table, data = build_table()
    print(table)
    p = plot_global_summary(data)
    print("\nwrote", p)
    print("wrote", RESULTS_DIR / "master_table.md")


if __name__ == "__main__":
    main()
