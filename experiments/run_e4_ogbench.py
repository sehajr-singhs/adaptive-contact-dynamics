"""Run the three controllers on the REAL OGBench cube-single env. Runs in .venv-ogbench (Py 3.11/3.12).

Uses experiments.run_env._rollout_arm unchanged, so the controllers, metrics, and per-seed JSON
schema are identical to every other environment. Writes results/e4_cube/<controller>/<regime>/seedN.json.
"""
from __future__ import annotations
import argparse
import numpy as np

from src.utils import load_config
from experiments.run_env import _rollout_arm, CONTROLLERS
from src.envs.ogbench_cube import OGBenchCubeArm


def rollout(controller, regime, seed, cfg, ref_bound=None, ref_bound_post=None, save=True):
    env = OGBenchCubeArm(dt=cfg["sim"]["dt"], substeps=cfg["sim"]["substeps"],
                         nominal_mass=cfg["sim"].get("nominal_mass", 0.5),
                         nominal_friction=cfg["sim"].get("nominal_friction", 0.5),
                         push_dist=cfg["sim"].get("push_dist", 0.26))
    env.episode_T = cfg["sim"]["episode_seconds"]
    return _rollout_arm("e4_cube", env, controller, regime, seed, cfg,
                        quick=False, ref_bound=ref_bound, ref_bound_post=ref_bound_post, save=save)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="*", default=["step", "continuous"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    cfg = load_config("e4_cube")
    ordered = ["fixed_baseline", "constrained_adaptive", "unconstrained_adaptive"]
    for rg in args.regimes:
        for s in args.seeds:
            ref_bound = ref_bound_post = None
            for c in ordered:
                m = rollout(c, rg, s, cfg, ref_bound, ref_bound_post, save=True)
                if c == "fixed_baseline":
                    ref_bound = m["energy_bound_own"]
                    ref_bound_post = m["energy_bound_post_own"]
                rs = m["recovery_seconds"]
                rs = f"{rs:.2f}s" if rs is not None else "no-recover"
                v = m["lyap_violation_frac"]
                print(f"  [e4_cube/OGBench] {c:24s} {rg:10s} seed{s} "
                      f"recover={rs} post={m['ss_error_post_deg']:.2f}cm viol={v:.2f}")


if __name__ == "__main__":
    main()
