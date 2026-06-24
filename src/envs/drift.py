"""Drift injection shared across environments.

A DriftSchedule reads the per-environment entry from configs/drift_schedules.yaml and,
at the scheduled time, edits the *live* MuJoCo model in place (body mass + inertia, joint
damping, geom friction, actuator gain). The nominal twin used for computed-torque feedback
is never touched, so the edit shows up as a matched uncertainty the controller must adapt to.

Two regimes:
  - step:       one abrupt multiplicative change at trigger_frac of the episode.
  - continuous: the same target change ramped linearly over [start_frac, end_frac].
"""
from __future__ import annotations
import numpy as np
import mujoco


class DriftSchedule:
    def __init__(self, env, spec: dict, regime: str, n_steps: int):
        self.env = env
        self.model = env.model
        self.regime = regime
        self.n_steps = n_steps
        self.spec = spec.get(regime, {})
        self._applied_step = False
        self._cache_baselines()
        if regime == "step":
            self.trigger = int(self.spec.get("trigger_frac", 0.5) * n_steps)
        else:
            self.start = int(self.spec.get("start_frac", 0.3) * n_steps)
            self.end = int(self.spec.get("end_frac", 0.9) * n_steps)
        self.drift_step = self.trigger if regime == "step" else self.start

    def _cache_baselines(self):
        """Snapshot the original parameter values so ramps interpolate from them."""
        m = self.model
        self.base_body_mass = m.body_mass.copy()
        self.base_body_inertia = m.body_inertia.copy()
        self.base_dof_damping = m.dof_damping.copy()
        self.base_dof_armature = m.dof_armature.copy()
        self.base_dof_frictionloss = m.dof_frictionloss.copy()
        self.base_geom_friction = m.geom_friction.copy()
        self.base_gainprm = m.actuator_gainprm.copy()

    def _set_fraction(self, frac: float):
        """Apply `frac` of the full target drift (frac in [0,1])."""
        m = self.model
        env = self.env
        # body mass + inertia multipliers
        for bname, mult in self.spec.get("body_mass", {}).items():
            try:
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
            except Exception:
                continue
            if bid < 0:
                continue
            f = 1.0 + frac * (mult - 1.0)
            m.body_mass[bid] = self.base_body_mass[bid] * f
            m.body_inertia[bid] = self.base_body_inertia[bid] * f
        # joint damping multipliers
        for jname, mult in self.spec.get("dof_damping", {}).items():
            try:
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
                dof = m.jnt_dofadr[jid]
            except Exception:
                continue
            f = 1.0 + frac * (mult - 1.0)
            m.dof_damping[dof] = self.base_dof_damping[dof] * f
        # joint armature (reflected motor/gearbox inertia) multipliers
        for jname, mult in self.spec.get("dof_armature", {}).items():
            try:
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
                dof = m.jnt_dofadr[jid]
            except Exception:
                continue
            f = 1.0 + frac * (mult - 1.0)
            m.dof_armature[dof] = self.base_dof_armature[dof] * f
        # joint Coulomb friction (frictionloss) multipliers -> "table friction" for the pusher
        for jname, mult in self.spec.get("dof_frictionloss", {}).items():
            try:
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
                dof = m.jnt_dofadr[jid]
            except Exception:
                continue
            f = 1.0 + frac * (mult - 1.0)
            m.dof_frictionloss[dof] = self.base_dof_frictionloss[dof] * f
        if "frictionloss_mult" in self.spec:
            mult = self.spec["frictionloss_mult"]
            f = 1.0 + frac * (mult - 1.0)
            m.dof_frictionloss[:] = self.base_dof_frictionloss * f
        # global geom friction (tangential component) multiplier
        if "geom_friction_mult" in self.spec:
            mult = self.spec["geom_friction_mult"]
            f = 1.0 + frac * (mult - 1.0)
            m.geom_friction[:, 0] = self.base_geom_friction[:, 0] * f
        # object mass multiplier (named body "object" / "cube" if present)
        if "object_mass_mult" in self.spec:
            mult = self.spec["object_mass_mult"]
            f = 1.0 + frac * (mult - 1.0)
            for cand in ("object", "cube", "obj", "object_0"):
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cand)
                if bid is not None and bid >= 0:
                    m.body_mass[bid] = self.base_body_mass[bid] * f
                    m.body_inertia[bid] = self.base_body_inertia[bid] * f
        # actuator gain multiplier
        if "actuator_gain" in self.spec:
            mult = self.spec["actuator_gain"]
            f = 1.0 + frac * (mult - 1.0)
            m.actuator_gainprm[:, 0] = self.base_gainprm[:, 0] * f

    def update(self, k: int) -> bool:
        """Call once per control step. Returns True if any drift is active now."""
        if not self.spec:
            return False
        if self.regime == "step":
            if k >= self.trigger and not self._applied_step:
                self._set_fraction(1.0)
                self._applied_step = True
            return self._applied_step
        else:
            if k < self.start:
                return False
            frac = min(1.0, (k - self.start) / max(1, self.end - self.start))
            self._set_fraction(frac)
            return True
