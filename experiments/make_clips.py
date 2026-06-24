"""Re-render rollout clips from saved trajectories (no re-simulation)."""
from __future__ import annotations
import argparse
import numpy as np

from src.utils import RESULTS_DIR, CLIPS_DIR, ensure_dir
from src.viz import render
from src.envs.planar_arm import XML as E1_XML
from src.envs.push_drift import XML as E3_XML_TMPL
from src.envs.cube_drift import XML as E4_XML

ENV_XML = {
    "e1_planar_arm": E1_XML,
    "e3_push": E3_XML_TMPL.format(fl=1.5),
    "e4_cube": E4_XML,
}


def _load(env, controller, regime):
    p = RESULTS_DIR / env / "trajectories" / f"{controller}_{regime}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["qpos"], int(d["drift_step"])


def clips_for_env(env, regime="step"):
    xml = ENV_XML.get(env)
    if xml is None:
        return []
    out = ensure_dir(CLIPS_DIR / env)
    made = []
    base = _load(env, "fixed_baseline", regime)
    con = _load(env, "constrained_adaptive", regime)
    unc = _load(env, "unconstrained_adaptive", regime)
    if base is not None:
        p, _ = render.render_trajectory(xml, base[0], base[1], out / f"{regime}_baseline",
                                        label="Fixed baseline")
        made.append(p)
    if con is not None:
        p, _ = render.render_trajectory(xml, con[0], con[1], out / f"{regime}_constrained",
                                        label="Constrained adaptive")
        made.append(p)
    if base is not None and con is not None:
        p = render.side_by_side(xml, base[0], con[0], base[1], out / f"{regime}_sidebyside",
                                label_a="Fixed baseline", label_b="Constrained adaptive")
        made.append(p)
    if con is not None and unc is not None:
        p = render.side_by_side(xml, unc[0], con[0], con[1], out / f"{regime}_stability",
                                label_a="Unconstrained (ablation)", label_b="Constrained")
        made.append(p)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="*", default=["e1_planar_arm"])
    ap.add_argument("--regime", default="step")
    args = ap.parse_args()
    for env in args.envs:
        for p in clips_for_env(env, args.regime):
            print("  wrote", p)


if __name__ == "__main__":
    main()
