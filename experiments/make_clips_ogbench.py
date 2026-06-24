"""Render E4 clips from the REAL OGBench cube-single rollouts. Runs in .venv-ogbench.

Replays the saved full-model qpos trajectories (written by _rollout_arm for seed 0) into the real
OGBench model and renders offscreen, so the clips are the actual benchmark scene.
"""
from __future__ import annotations
import numpy as np
import mujoco
import imageio.v2 as imageio
import ogbench

from src.utils import RESULTS_DIR, CLIPS_DIR, ensure_dir
from src.viz.render import _text_overlay

ENV_ID = "cube-single-play-singletask-v0"


def _model():
    env = ogbench.make_env_and_datasets(ENV_ID, env_only=True)
    m = env.unwrapped.model
    m.vis.global_.offwidth = 480
    m.vis.global_.offheight = 360
    return m


def _camera():
    # frames the cube's push lane and the UR5e end effector pushing it across the table
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.42, 0.02, 0.05]
    cam.distance = 0.72
    cam.elevation = -22
    cam.azimuth = 150
    return cam


def render_traj(model, qpos, drift_step, out_path, label, fps=50, stride=2):
    data = mujoco.MjData(model)
    cam = _camera()
    renderer = mujoco.Renderer(model, 360, 480)
    frames = []
    for k in range(0, len(qpos), stride):
        data.qpos[:] = qpos[k][:model.nq]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        f = _text_overlay(renderer.render(), label, drift=(k >= drift_step))
        frames.append(f)
    renderer.close()
    ensure_dir(out_path.parent)
    imageio.mimsave(out_path.with_suffix(".mp4"), frames, fps=fps, macro_block_size=1)
    imageio.mimsave(out_path.with_suffix(".gif"), frames[::2], fps=fps // 2, loop=0)
    return frames


def load(controller, regime="step"):
    p = RESULTS_DIR / "e4_cube" / "trajectories" / f"{controller}_{regime}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["qpos"], int(d["drift_step"])


def main():
    m = _model()
    out = ensure_dir(CLIPS_DIR / "e4_cube")
    base = load("fixed_baseline"); con = load("constrained_adaptive"); unc = load("unconstrained_adaptive")
    fa = render_traj(m, base[0], base[1], out / "step_baseline", "Fixed baseline") if base else None
    fc = render_traj(m, con[0], con[1], out / "step_constrained", "Constrained adaptive") if con else None
    fu = render_traj(m, unc[0], unc[1], out / "step_stability_unc", "Unconstrained (ablation)") if unc else None
    # side-by-side baseline vs constrained
    if fa and fc:
        n = min(len(fa), len(fc)); sep = np.full((fa[0].shape[0], 4, 3), 255, np.uint8)
        combo = [np.concatenate([fa[i], sep, fc[i]], 1) for i in range(n)]
        imageio.mimsave(out / "step_sidebyside.mp4", combo, fps=50, macro_block_size=1)
        imageio.mimsave(out / "step_sidebyside.gif", combo[::2], fps=25, loop=0)
    if fu and fc:
        n = min(len(fu), len(fc)); sep = np.full((fu[0].shape[0], 4, 3), 255, np.uint8)
        combo = [np.concatenate([fu[i], sep, fc[i]], 1) for i in range(n)]
        imageio.mimsave(out / "step_stability.mp4", combo, fps=50, macro_block_size=1)
        imageio.mimsave(out / "step_stability.gif", combo[::2], fps=25, loop=0)
    print("wrote E4 OGBench clips to", out)


if __name__ == "__main__":
    main()
