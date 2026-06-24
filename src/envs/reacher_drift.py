"""E2: the gymnasium Reacher arm, driven by our own MRAC control loop.

Reacher is a 2-joint torque-controlled arm (plus two unactuated slide joints that only position
the visual target), so the same computed-torque MRAC backbone from E1 applies directly. We bypass
the gym reward/step machinery and step the underlying MuJoCo model ourselves, which lets us inject
drift (actuator gain down, distal-link inertia up) the same way as every other environment. Using
the real Reacher model means a reader who knows the benchmark sees an immediate point of comparison.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
import mujoco

ARM_DOFS = [0, 1]  # the two actuated hinge joints; dofs 2,3 are the static target slides


def _reacher_model():
    env = gym.make("Reacher-v5")
    m = env.unwrapped.model
    env.close()
    return m


class ReacherArm:
    def __init__(self, dt: float = 0.01, substeps: int = 2):
        # Build from a fresh spec so live and nominal are independent MjModel objects.
        self.model = _reacher_model()
        self.nominal = _reacher_model()
        self.data = mujoco.MjData(self.model)
        self.ndata = mujoco.MjData(self.nominal)
        self.model.opt.timestep = dt / substeps
        self.nominal.opt.timestep = dt / substeps
        self.dt = dt
        self.substeps = substeps
        self.nu = len(ARM_DOFS)
        self.nq = self.model.nq
        self._M = np.zeros((self.model.nv, self.model.nv))

    def reset(self, q0=None, qd0=None):
        mujoco.mj_resetData(self.model, self.data)
        # fix the target at a constant location (it is not part of the control task)
        self.data.qpos[2] = 0.1
        self.data.qpos[3] = 0.0
        if q0 is not None:
            self.data.qpos[ARM_DOFS] = q0
        if qd0 is not None:
            self.data.qvel[ARM_DOFS] = qd0
        mujoco.mj_forward(self.model, self.data)
        return self.q, self.qd

    @property
    def q(self):
        return self.data.qpos[ARM_DOFS].copy()

    @property
    def qd(self):
        return self.data.qvel[ARM_DOFS].copy()

    def full_qpos(self):
        return self.data.qpos.copy()

    def step(self, tau):
        self.data.ctrl[:] = np.asarray(tau, dtype=float)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        return self.q, self.qd

    def nominal_dynamics(self, q, qd):
        self.ndata.qpos[ARM_DOFS] = q
        self.ndata.qvel[ARM_DOFS] = qd
        mujoco.mj_forward(self.nominal, self.ndata)
        mujoco.mj_fullM(self.nominal, self._M, self.ndata.qM)
        idx = np.ix_(ARM_DOFS, ARM_DOFS)
        M = self._M[idx].copy()
        b = self.ndata.qfrc_bias[ARM_DOFS].copy()
        return M, b
