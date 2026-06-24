"""One command to reproduce everything.

    python -m experiments.run_all --quick   # smoke-test: 1 seed, short episodes, every env + figures
    python -m experiments.run_all --full    # the paper numbers: all seeds, both regimes, + certificate,
                                            # gain sweeps, figures, master table, clips

Both actually run end to end on CPU and write real artifacts to disk.
"""
from __future__ import annotations
import argparse
import copy
import json
import numpy as np

from src.utils import load_config, RESULTS_DIR, ensure_dir
from experiments.run_env import run_env, ROLLOUTS, CONTROLLERS

ENVS = ["e1_planar_arm", "e2_reacher", "e3_push", "e4_cube"]


def _gain_sweep(env, gammas, seeds=(0, 1, 2)):
    cfg = load_config(env)
    fn = ROLLOUTS[env]
    out = {"gammas": list(gammas),
           "constrained_adaptive": {"mean": [], "std": []},
           "unconstrained_adaptive": {"mean": [], "std": []}}
    ceiling = None
    for g in gammas:
        c = copy.deepcopy(cfg)
        c["adaptive"]["gamma"] = g
        for ctrl in ["constrained_adaptive", "unconstrained_adaptive"]:
            vals = [fn(ctrl, "step", s, c, save=False)["ss_error_post_deg"] for s in seeds]
            out[ctrl]["mean"].append(float(np.mean(vals)))
            out[ctrl]["std"].append(float(np.std(vals)))
        pre = [fn("unconstrained_adaptive", "step", s, c, save=False)["ss_error_pre_deg"]
               for s in seeds]
        if np.mean(pre) < 5.0:
            ceiling = g
    out["ceiling"] = ceiling * 1.4 if ceiling else None
    json.dump(out, open(ensure_dir(RESULTS_DIR / env) / "gain_sweep.json", "w"), indent=2)
    print(f"  gain sweep {env}: ceiling~{ceiling}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--env", default=None, help="run a single environment")
    args = ap.parse_args()
    quick = args.quick or not args.full
    envs = [args.env] if args.env else ENVS

    regimes = ["step"] if quick else ["step", "continuous"]
    for env in envs:
        cfg = load_config(env)
        seeds = cfg["run"]["quick_seeds"] if quick else cfg["run"]["seeds"]
        print(f"== {env} ({'quick' if quick else 'full'}) ==")
        run_env(env, CONTROLLERS, regimes, seeds, quick=quick)

    if not quick:
        # gain sweeps (the stability-ceiling figures) for the two nonlinear arms
        print("== gain sweeps ==")
        _gain_sweep("e1_planar_arm", [50, 100, 200, 400, 700, 1000, 2000, 4000])
        _gain_sweep("e4_cube", [100, 300, 700, 1500, 2500, 4000])
        # neural Lyapunov certificate on E1
        print("== certificate (E1) ==")
        from experiments import run_certificate
        run_certificate.main("e1_planar_arm")

    # figures + master table always
    print("== figures + summary ==")
    from experiments import make_figures, make_summary
    from src.viz import diagrams
    fig_regimes = tuple(regimes)
    for env in envs:
        seeds = load_config(env)["run"]["quick_seeds" if quick else "seeds"]
        for f in make_figures.figures_for_env(env, seeds, regimes=fig_regimes):
            print("  wrote", f)
    diagrams.architecture()
    _, data = make_summary.build_table()
    make_summary.plot_global_summary(data)

    if not quick:
        print("== clips ==")
        from experiments import make_clips
        for env in ["e1_planar_arm", "e3_push", "e4_cube"]:
            make_clips.clips_for_env(env, "step")
    print("done.")


if __name__ == "__main__":
    main()
