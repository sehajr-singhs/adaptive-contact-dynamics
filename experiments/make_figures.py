"""Regenerate every figure from saved results JSON, no re-simulation."""
from __future__ import annotations
import argparse
import json

from src.utils import RESULTS_DIR, load_config
from src.viz import plots
from src.viz import certificate_plot

ENVS = ["e1_planar_arm", "e2_reacher", "e3_push", "e4_cube"]


def figures_for_env(env, seeds, regimes=("step", "continuous")):
    made = []
    have = (RESULTS_DIR / env / "fixed_baseline").exists()
    if not have:
        return made
    for rg in regimes:
        if (RESULTS_DIR / env / "constrained_adaptive" / rg).exists():
            made.append(plots.plot_tracking_error(env, rg, seeds))
            made.append(plots.plot_energy(env, rg, seeds))
    made.append(plots.plot_recovery_bar(env, list(regimes), seeds))
    sweep = RESULTS_DIR / env / "gain_sweep.json"
    if sweep.exists():
        made.append(plots.plot_gain_sweep(sweep, outname=f"{env}_gain_sweep.png"))
    cert = RESULTS_DIR / env / "certificate.json"
    if cert.exists():
        made.append(certificate_plot.plot_certificate(cert, outname=f"{env}_certificate.png"))
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="*", default=ENVS)
    args = ap.parse_args()
    for env in args.envs:
        cfgp = (RESULTS_DIR / env)
        if not cfgp.exists():
            continue
        cfg = load_config(env)
        seeds = cfg["run"]["seeds"]
        made = figures_for_env(env, seeds)
        for m in made:
            print("  wrote", m)


if __name__ == "__main__":
    main()
