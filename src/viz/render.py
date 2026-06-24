"""Render MuJoCo rollouts to mp4/gif from saved trajectories (no re-simulation).

Trajectories are replayed into the model purely for visualization: we set qpos frame by frame
and render offscreen. A minimal on-frame label shows the environment and controller, and a DRIFT
banner appears at the drift event. Side-by-side composition time-syncs two rollouts.
"""
from __future__ import annotations
import numpy as np
import mujoco
import imageio.v2 as imageio

from ..utils import CLIPS_DIR, ensure_dir


try:
    from PIL import Image, ImageDraw, ImageFont
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False


def _text_overlay(frame, label="", drift=False):
    """Top label bar + a DRIFT banner/border after the drift event."""
    img = frame.copy()
    h, w, _ = img.shape
    bar = int(0.12 * h)
    img[:bar] = (img[:bar] * 0.25).astype(img.dtype)  # darken top bar for contrast
    if drift:
        b = max(2, int(0.012 * h))
        img[:b] = [200, 40, 40]; img[-b:] = [200, 40, 40]
        img[:, :b] = [200, 40, 40]; img[:, -b:] = [200, 40, 40]
    if _HAVE_PIL and (label or drift):
        pim = Image.fromarray(img)
        d = ImageDraw.Draw(pim)
        try:
            font = ImageFont.truetype("arial.ttf", int(0.07 * h))
        except Exception:
            font = ImageFont.load_default()
        if label:
            d.text((6, 3), label, fill=(255, 255, 255), font=font)
        if drift:
            d.text((w - int(0.30 * w), 3), "DRIFT", fill=(255, 120, 120), font=font)
        img = np.asarray(pim)
    return img


def render_trajectory(model_xml, qpos_traj, drift_step, out_path, fps=50,
                      width=360, height=300, stride=2, label=""):
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    frames = []
    for k in range(0, len(qpos_traj), stride):
        data.qpos[:] = qpos_traj[k][: model.nq]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        frame = renderer.render()
        frame = _text_overlay(frame, label, drift=(k >= drift_step))
        frames.append(frame)
    renderer.close()
    ensure_dir(out_path.parent)
    imageio.mimsave(out_path.with_suffix(".mp4"), frames, fps=fps, macro_block_size=1)
    imageio.mimsave(out_path.with_suffix(".gif"), frames[::2], fps=fps // 2, loop=0)
    return out_path.with_suffix(".mp4"), frames


def side_by_side(model_xml, traj_a, traj_b, drift_step, out_path,
                 label_a="", label_b="", fps=50, width=360, height=300, stride=2):
    _, fa = render_trajectory(model_xml, traj_a, drift_step,
                              out_path.parent / "_tmp_a", fps, width, height, stride, label_a)
    _, fb = render_trajectory(model_xml, traj_b, drift_step,
                              out_path.parent / "_tmp_b", fps, width, height, stride, label_b)
    n = min(len(fa), len(fb))
    sep = np.full((fa[0].shape[0], 4, 3), 255, dtype=fa[0].dtype)
    combo = [np.concatenate([fa[i], sep, fb[i]], axis=1) for i in range(n)]
    ensure_dir(out_path.parent)
    imageio.mimsave(out_path.with_suffix(".mp4"), combo, fps=fps, macro_block_size=1)
    imageio.mimsave(out_path.with_suffix(".gif"), combo[::2], fps=fps // 2, loop=0)
    # clean temp files
    for stem in ("_tmp_a", "_tmp_b"):
        for ext in (".mp4", ".gif"):
            p = (out_path.parent / stem).with_suffix(ext)
            if p.exists():
                p.unlink()
    return out_path.with_suffix(".mp4")
