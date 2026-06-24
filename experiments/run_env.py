"""Run a single environment x controller x drift-regime x seed and log real metrics.

Currently dispatches E1 (planar arm). E2-E4 register their own rollout functions here as they
come online. Every call writes a per-seed JSON to results/ and (for seed 0) a trajectory npz to
results/<env>/trajectories/ so clips regenerate without re-simulating.
"""
from __future__ import annotations
import argparse
import numpy as np

from src.utils import load_config, set_seed, ensure_dir, RESULTS_DIR
from src.metrics.logging import save_result
from src.metrics import recovery as R
from src.metrics import stability as S
from src.adaptive.mrac import ReferenceModel, build_error_system
from src.adaptive.stable_nn import FeatureMap, ConstrainedAdaptive
from src.adaptive.unconstrained_nn import UnconstrainedAdaptive
from src.controllers.pd_computed_torque import FixedBaseline, computed_torque
from src.envs.planar_arm import PlanarArm
from src.envs.drift import DriftSchedule

CONTROLLERS = ["fixed_baseline", "constrained_adaptive", "unconstrained_adaptive"]


def _make_controller(name, feature, n, P, B, dt, a):
    if name == "fixed_baseline":
        return FixedBaseline(n)
    if name == "constrained_adaptive":
        return ConstrainedAdaptive(feature, n, P, B, dt,
                                   gamma=a["gamma"], sigma=a["sigma_mod"],
                                   theta_max=a["theta_max"])
    if name == "unconstrained_adaptive":
        return UnconstrainedAdaptive(feature, n, P, B, dt, gamma=a["gamma"])
    raise ValueError(name)


def _rollout_arm(env_key, env, controller_name, regime, seed, cfg, quick=False,
                 ref_bound=None, ref_bound_post=None, save=True):
    """Generic MRAC joint-tracking rollout for any torque-controlled arm adapter.

    The adapter must expose: nu, reset(q0, qd0), q, qd, step(tau), nominal_dynamics(q, qd).
    """
    set_seed(seed)
    sim = cfg["sim"]
    dt = sim["dt"]
    seconds = sim["quick_seconds"] if quick else sim["episode_seconds"]
    n_steps = int(seconds / dt)

    n = env.nu

    task = cfg["task"]
    rm = cfg["reference_model"]
    bl = cfg["baseline"]
    a = cfg["adaptive"]

    rng = np.random.default_rng(seed)
    ref = ReferenceModel(n, rm["wn"], rm["zeta"], dt)
    A_ref, B, P, Q = build_error_system(n, bl["kp"], bl["kd"], q_scale=a["lyap_Q"])
    # A different random feature draw per seed: shows the result is robust to the random
    # feature map, not an artifact of one lucky initialization. All controllers in a given
    # seed share the same draw so the ablation stays apples-to-apples.
    # Feature input: "q_qd" (default) uses [q, q_dot] (2n dims); "q" uses joint positions only (n
    # dims), which keeps the RBF grid tractable for higher-DOF arms where 2n is too large to tile.
    feat_input = a.get("feature_input", "q_qd")
    feat_dim = n if feat_input == "q" else 2 * n
    feature = FeatureMap(in_dim=feat_dim, per_dim=a.get("rbf_per_dim", 3),
                         seed=1000 + seed, region=a.get("rbf_region"))
    ctrl = _make_controller(controller_name, feature, n, P, B, dt, a)
    drift = DriftSchedule(env, cfg["drift"], regime, n_steps)

    # per-seed perturbed initial condition + light actuation noise -> genuine seed variance
    # (scale is rad for the arm envs, m for the cube push, so it is configurable per env)
    init_perturb = float(cfg["run"].get("init_perturb", 0.05))
    off = np.array(task["cmd_offset"], dtype=float)
    q0 = off + rng.normal(0, init_perturb, n)
    qd0 = rng.normal(0, init_perturb, n)
    tau_noise = float(cfg.get("noise", {}).get("torque_std", 0.3))
    tau_clip = float(cfg.get("baseline", {}).get("tau_clip", 40.0))
    q_init, _ = env.reset(q0=q0, qd0=qd0)
    ref.reset(q_init)   # start the reference at the actual initial config (e.g. an IK-placed arm)

    freqs = np.array(task["cmd_freqs"])
    amps = np.array(task["cmd_amps"])

    err_log, eaug_log, tau_log, qpos_log = [], [], [], []
    has_cmd = hasattr(env, "command")
    for k in range(n_steps):
        t = k * dt
        # An env may supply its own command trajectory (e.g. the OGBench arm's IK push path);
        # otherwise use the default per-joint sinusoid.
        x_cmd = env.command(t) if has_cmd else off + amps * np.sin(2 * np.pi * freqs * t)
        x_ref, xd_ref, xdd_ref = ref.step(x_cmd)

        q, qd = env.q, env.qd
        e = q - x_ref
        ed = qd - xd_ref
        e_aug = np.concatenate([e, ed])
        state = q if feat_input == "q" else np.concatenate([q, qd])

        u_ad = ctrl.u_ad(state)
        M0, b0 = env.nominal_dynamics(q, qd)
        tau = computed_torque(M0, b0, xdd_ref, e, ed, bl["kp"], bl["kd"], u_ad)
        tau = tau + rng.normal(0, tau_noise, n)   # actuation noise
        tau = np.clip(tau, -tau_clip, tau_clip)

        drift.update(k)          # mutate the live plant on schedule
        env.step(tau)
        ctrl.update(state, e_aug)

        err_log.append(e.copy())
        eaug_log.append(e_aug.copy())
        tau_log.append(tau.copy())
        qpos_log.append(env.full_qpos())

    err_log = np.array(err_log)
    err_scale = float(cfg["run"].get("error_scale", 180.0 / np.pi))
    err_unit = cfg["run"].get("error_unit", "deg")
    err_deg = R.tracking_error(err_log, err_scale)   # display-unit error series
    E_series = S.error_energy(eaug_log, P)
    drift_step = drift.drift_step
    thr = cfg["run"]["threshold_deg"]

    # Shared "healthy" reference bound (the tuned baseline's nominal tracking energy) so a
    # controller that is already unstable before the drift is correctly flagged rather than
    # normalized against its own blown-up pre-drift energy.
    own_bound = S._pre_bound(E_series, drift_step)
    own_post_bound = S.post_bound(E_series, drift_step)
    bound = ref_bound if ref_bound is not None else own_bound          # recovery reference (nominal)
    inst_bound = ref_bound_post if ref_bound_post is not None else own_post_bound  # instability ref
    viol, frac, _ = S.violation_fraction(E_series, drift_step, ref_bound=inst_bound)
    rec_steps, rec_sec = R.recovery_time_energy(E_series, drift_step, dt, bound)
    rec_steps_abs, rec_sec_abs = R.recovery_time(err_deg, drift_step, dt, thr)
    pre_lo, pre_hi = max(0, drift_step - 50), drift_step
    post_lo, post_hi = max(0, n_steps - 50), n_steps
    metrics = {
        "env": env_key,
        "controller": controller_name,
        "regime": regime,
        "seed": seed,
        "dt": dt,
        "n_steps": n_steps,
        "drift_step": drift_step,
        "threshold_deg": thr,
        "error_unit": err_unit,
        "recovery_steps": rec_steps,
        "recovery_seconds": rec_sec,
        "recovery_seconds_abs": rec_sec_abs,
        "ss_error_pre_deg": R.steady_state_error(err_deg, pre_lo, pre_hi),
        "ss_error_post_deg": R.steady_state_error(err_deg, post_lo, post_hi),
        "max_error_post_deg": float(np.max(err_deg[drift_step:])),
        "control_effort": R.control_effort(tau_log, dt),
        "lyap_violations": None,
        "lyap_violation_frac": None,
        "lyap_divergence": None,
        "err_deg_series": err_deg.tolist(),
        "energy_series": [float(v) for v in E_series],
    }
    metrics["lyap_violations"] = viol
    metrics["lyap_violation_frac"] = frac
    metrics["lyap_divergence"] = S.divergence(E_series, drift_step, ref_bound=inst_bound)
    metrics["energy_bound"] = float(bound)
    metrics["energy_bound_own"] = float(own_bound)
    metrics["energy_bound_post_own"] = float(own_post_bound)
    metrics["stable_pre_drift"] = bool(metrics["ss_error_pre_deg"] < 5.0)
    if hasattr(env, "cube_to_goal_cm"):
        metrics["cube_to_goal_cm"] = env.cube_to_goal_cm()

    if not save:
        return metrics
    save_result(env_key, controller_name, regime, seed, metrics)

    # store trajectory for clips (seed 0 only, keeps repo small)
    if seed == 0:
        tdir = ensure_dir(RESULTS_DIR / env_key / "trajectories")
        np.savez_compressed(
            tdir / f"{controller_name}_{regime}.npz",
            qpos=np.array(qpos_log),
            err_deg=err_deg, drift_step=drift_step, dt=dt,
        )
    return metrics


def rollout_e1(controller_name, regime, seed, cfg, quick=False, ref_bound=None, ref_bound_post=None, save=True):
    env = PlanarArm(dt=cfg["sim"]["dt"], substeps=cfg["sim"]["substeps"])
    return _rollout_arm("e1_planar_arm", env, controller_name, regime, seed, cfg, quick,
                        ref_bound, ref_bound_post, save)


def rollout_e2(controller_name, regime, seed, cfg, quick=False, ref_bound=None, ref_bound_post=None, save=True):
    from src.envs.reacher_drift import ReacherArm
    env = ReacherArm(dt=cfg["sim"]["dt"], substeps=cfg["sim"]["substeps"])
    return _rollout_arm("e2_reacher", env, controller_name, regime, seed, cfg, quick,
                        ref_bound, ref_bound_post, save)


def rollout_e3(controller_name, regime, seed, cfg, quick=False, ref_bound=None, ref_bound_post=None, save=True):
    from src.envs.push_drift import Block2D
    env = Block2D(dt=cfg["sim"]["dt"], substeps=cfg["sim"]["substeps"],
                  frictionloss=cfg["sim"].get("frictionloss", 1.5))
    return _rollout_arm("e3_push", env, controller_name, regime, seed, cfg, quick,
                        ref_bound, ref_bound_post, save)


def rollout_e4(controller_name, regime, seed, cfg, quick=False, ref_bound=None, ref_bound_post=None, save=True):
    from src.envs.cube_drift import CarryArm
    env = CarryArm(dt=cfg["sim"]["dt"], substeps=cfg["sim"]["substeps"])
    return _rollout_arm("e4_cube", env, controller_name, regime, seed, cfg, quick,
                        ref_bound, ref_bound_post, save)


ROLLOUTS = {"e1_planar_arm": rollout_e1, "e2_reacher": rollout_e2,
            "e3_push": rollout_e3, "e4_cube": rollout_e4}


def run_env(env_key, controllers, regimes, seeds, quick=False):
    cfg = load_config(env_key)
    fn = ROLLOUTS[env_key]
    summary = {}
    # Order so the tuned baseline runs first per (regime, seed); its nominal tracking energy
    # is the shared healthy bound the adaptive controllers are scored against.
    ordered = (["fixed_baseline"] if "fixed_baseline" in controllers else []) + \
              [c for c in controllers if c != "fixed_baseline"]
    for rg in regimes:
        for s in seeds:
            ref_bound = None
            ref_bound_post = None
            for c in ordered:
                m = fn(c, rg, s, cfg, quick=quick, ref_bound=ref_bound,
                       ref_bound_post=ref_bound_post)
                if c == "fixed_baseline":
                    ref_bound = m["energy_bound_own"]
                    ref_bound_post = m["energy_bound_post_own"]
                rs = m["recovery_seconds"]
                rs_str = f"{rs:.2f}s" if rs is not None else "no-recover"
                viol = m["lyap_violation_frac"]
                viol_str = f" viol={viol:.2f}" if viol is not None else ""
                print(f"  [{env_key}] {c:24s} {rg:10s} seed{s} "
                      f"recover={rs_str} post_err={m['ss_error_post_deg']:.2f}deg{viol_str}")
                summary[(c, rg, s)] = m
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="e1_planar_arm")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--controllers", nargs="*", default=CONTROLLERS)
    ap.add_argument("--regimes", nargs="*", default=["step", "continuous"])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.env)
    seeds = args.seeds if args.seeds is not None else (
        cfg["run"]["quick_seeds"] if args.quick else cfg["run"]["seeds"])
    run_env(args.env, args.controllers, args.regimes, seeds, quick=args.quick)
