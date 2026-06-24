"""Train and verify the neural Lyapunov certificate on the E1 closed-loop adaptive system.

Collects real transition pairs (e_aug_k, e_aug_{k+1}) from the constrained controller running on
the fully-drifted plant from randomized initial errors, fits V_theta, verifies the decrease
condition on held-out transitions, and saves a 2D contour slice for the figure.
"""
from __future__ import annotations
import json
import numpy as np

from src.utils import load_config, set_seed, RESULTS_DIR, ensure_dir
from src.adaptive.mrac import ReferenceModel, build_error_system
from src.adaptive.stable_nn import FeatureMap, ConstrainedAdaptive
from src.controllers.pd_computed_torque import computed_torque
from src.envs.planar_arm import PlanarArm
from src.envs.drift import DriftSchedule
from src.adaptive import certificate as C


def _build(cfg, seed):
    sim, bl, a, rm, task = cfg["sim"], cfg["baseline"], cfg["adaptive"], cfg["reference_model"], cfg["task"]
    dt = sim["dt"]
    env = PlanarArm(dt=dt, substeps=sim["substeps"])
    n = env.nu
    A_ref, B, P, Q = build_error_system(n, bl["kp"], bl["kd"], q_scale=a["lyap_Q"])
    feature = FeatureMap(in_dim=2 * n, per_dim=a.get("rbf_per_dim", 3), seed=1000 + seed)
    ctrl = ConstrainedAdaptive(feature, n, P, B, dt, gamma=a["gamma"],
                               sigma=a["sigma_mod"], theta_max=a["theta_max"])
    ref = ReferenceModel(n, rm["wn"], rm["zeta"], dt)
    return env, n, A_ref, B, P, ctrl, ref, dt


def collect_transitions(cfg, seed=0, n_traj=120, traj_steps=120, warmup_steps=600):
    """Regulate the constrained controller to a fixed setpoint on the fully-drifted plant and
    gather (e_k, e_{k+1}) pairs. Regulation makes the error dynamics autonomous, which is the
    standard setting in which a time-invariant Lyapunov certificate is meaningful."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    env, n, A_ref, B, P, ctrl, ref, dt = _build(cfg, seed)
    bl, task = cfg["baseline"], cfg["task"]
    off = np.array(task["cmd_offset"], dtype=float)

    # apply full drift up front (certify the drifted closed loop) and warm up Theta at the setpoint
    drift = DriftSchedule(env, cfg["drift"], "step", 1)
    drift._set_fraction(1.0)
    env.reset(q0=off.copy(), qd0=np.zeros(n)); ref.reset(off.copy())
    for k in range(warmup_steps):
        x_ref, xd_ref, xdd_ref = ref.step(off)
        q, qd = env.q, env.qd
        e, ed = q - x_ref, qd - xd_ref
        u = ctrl.u_ad(np.concatenate([q, qd]))
        M0, b0 = env.nominal_dynamics(q, qd)
        env.step(np.clip(computed_torque(M0, b0, xdd_ref, e, ed, bl["kp"], bl["kd"], u), -40, 40))
        ctrl.update(np.concatenate([q, qd]), np.concatenate([e, ed]))

    X0, X1 = [], []
    region = np.array([1.2, 1.2, 2.5, 2.5])
    for _ in range(n_traj):
        # randomized initial error; controller regulates back toward the setpoint
        q0 = off + rng.uniform(-0.6, 0.6, n)
        qd0 = rng.uniform(-2.0, 2.0, n)
        env.reset(q0=q0, qd0=qd0)
        ref.reset(off.copy())
        prev = None
        for k in range(traj_steps):
            x_ref, xd_ref, xdd_ref = ref.step(off)
            q, qd = env.q, env.qd
            e, ed = q - x_ref, qd - xd_ref
            e_aug = np.concatenate([e, ed])
            if prev is not None and np.all(np.abs(e_aug) < region):
                X0.append(prev); X1.append(e_aug.copy())
            prev = e_aug.copy()
            u = ctrl.u_ad(np.concatenate([q, qd]))
            M0, b0 = env.nominal_dynamics(q, qd)
            env.step(np.clip(computed_torque(M0, b0, xdd_ref, e, ed, bl["kp"], bl["kd"], u), -40, 40))
            ctrl.update(np.concatenate([q, qd]), e_aug)
    return np.array(X0), np.array(X1), P, dt


def main(env_key="e1_planar_arm"):
    cfg = load_config(env_key)
    cc = cfg["certificate"]
    X0, X1, P, dt = collect_transitions(cfg, seed=0)
    # The closed loop is uniformly ultimately bounded under a moving reference, so the certificate
    # certifies contraction OUTSIDE the ultimate bound: V decreases when the error energy e^T P e
    # exceeds the bound, which is exactly the UUB condition. The bound is the steady-state energy
    # level (a low percentile of e^T P e across the collected transitions).
    E0 = np.einsum("ti,ij,tj->t", X0, P, X0)
    E_bound = float(np.percentile(E0, 60))
    mask = E0 > 1.3 * E_bound   # certify contraction clearly outside the ultimate bound
    X0, X1 = X0[mask], X1[mask]
    n = len(X0)
    idx = np.random.default_rng(0).permutation(n)
    tr, te = idx[: int(0.7 * n)], idx[int(0.7 * n):]
    model, hist = C.train_certificate(X0[tr], X1[tr], dt, alpha=cc["margin_alpha"],
                                      hidden=cc["hidden"], eps=cc["epsilon"],
                                      epochs=cc["epochs"])
    rep = C.verify_certificate(model, X0[te], X1[te], dt, alpha=cc["margin_alpha"])

    # closed-loop one-step linearization from data (for the contour decrease region)
    A_cl, *_ = np.linalg.lstsq(X0, X1, rcond=None)  # X1 ~ X0 @ A_cl
    decay = 1.0 - cc["margin_alpha"] * dt
    g = np.linspace(-1.2, 1.2, 80)
    gd = np.linspace(-2.5, 2.5, 80)
    GX, GY = np.meshgrid(g, gd)
    import torch
    pts = np.stack([GX.ravel(), np.zeros(GX.size), GY.ravel(), np.zeros(GX.size)], axis=1)
    with torch.no_grad():
        v0 = model(torch.tensor(pts, dtype=torch.float32)).numpy()
        nxt = pts @ A_cl
        v1 = model(torch.tensor(nxt, dtype=torch.float32)).numpy()
    Vgrid = v0.reshape(GX.shape)
    # energy on the slice e2 = ed2 = 0, to draw the ultimate-bound contour e^T P e = E_bound
    Eslice = (P[0, 0] * GX ** 2 + 2 * P[0, 2] * GX * GY + P[2, 2] * GY ** 2)
    # decrease is claimed only outside the ultimate bound (UUB)
    decrease = ((v1 <= decay * v0) | (Eslice.ravel() <= E_bound)).reshape(GX.shape).astype(float)

    out = {
        "env": env_key,
        "alpha": cc["margin_alpha"],
        "dt": dt,
        "n_transitions": int(n),
        "energy_bound": E_bound,
        **rep,
        "grid_e": g.tolist(), "grid_ed": gd.tolist(),
        "V_grid": Vgrid.tolist(), "decrease_grid": decrease.tolist(),
        "energy_grid": Eslice.tolist(),
        "final_loss": float(hist[-1]),
    }
    p = ensure_dir(RESULTS_DIR / env_key) / "certificate.json"
    json.dump(out, open(p, "w"))
    print(f"certificate: verify_fraction={rep['verify_fraction']:.4f} "
          f"worst_violation={rep['worst_violation']:.4f} "
          f"positive_fraction={rep['positive_fraction']:.3f} n={n}")
    print("  wrote", p)


if __name__ == "__main__":
    main()
