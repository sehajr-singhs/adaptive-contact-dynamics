"""E1: a 2-link planar arm in MuJoCo, the clean control testbed.

The arm hangs in a vertical plane and is torque-controlled at both joints, so gravity
plus coupling give a genuine nonlinear plant while staying small enough that the MRAC
math is legible and the certificate plots over a 2D state slice are readable.

We keep a second, never-drifted copy of the model ("nominal") to compute the
computed-torque feedback, so that any change to the live model shows up as a matched
uncertainty the adaptive term has to cancel.
"""
from __future__ import annotations
import numpy as np
import mujoco

XML = """
<mujoco model="planar2link">
  <option gravity="0 0 -9.81" integrator="RK4"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight diffuse="0.7 0.7 0.7"/>
  </visual>
  <worldbody>
    <light pos="0 -1 1.5" dir="0 1 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 -0.7" rgba="0.92 0.92 0.94 1"/>
    <site name="target" type="sphere" size="0.025" rgba="0.1 0.8 0.2 0.6" pos="0.4 0 0.1"/>
    <body name="link1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom name="g1" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"
            mass="1.0" rgba="0.20 0.45 0.85 1"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom name="g2" type="capsule" fromto="0 0 0 0.3 0 0" size="0.025"
              mass="0.7" rgba="0.90 0.50 0.20 1"/>
        <site name="ee" type="sphere" size="0.03" rgba="0.85 0.15 0.15 1" pos="0.3 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" gear="1"/>
    <motor joint="j2" gear="1"/>
  </actuator>
</mujoco>
"""


class PlanarArm:
    def __init__(self, dt: float = 0.01, substeps: int = 5):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        # Frozen nominal twin for computed-torque feedback.
        self.nominal = mujoco.MjModel.from_xml_string(XML)
        self.ndata = mujoco.MjData(self.nominal)
        self.model.opt.timestep = dt / substeps
        self.nominal.opt.timestep = dt / substeps
        self.dt = dt
        self.substeps = substeps
        self.nq = self.model.nq
        self.nu = self.model.nu
        self._M = np.zeros((self.nq, self.nq))

    def reset(self, q0=None, qd0=None):
        mujoco.mj_resetData(self.model, self.data)
        if q0 is not None:
            self.data.qpos[:] = q0
        if qd0 is not None:
            self.data.qvel[:] = qd0
        mujoco.mj_forward(self.model, self.data)
        return self.q, self.qd

    @property
    def q(self):
        return self.data.qpos.copy()

    @property
    def qd(self):
        return self.data.qvel.copy()

    def full_qpos(self):
        return self.data.qpos.copy()

    def step(self, tau):
        self.data.ctrl[:] = np.asarray(tau, dtype=float)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        return self.q, self.qd

    def nominal_dynamics(self, q, qd):
        """Return (M0, b0) from the never-drifted model: M0 q_ddot + b0 = tau."""
        self.ndata.qpos[:] = q
        self.ndata.qvel[:] = qd
        mujoco.mj_forward(self.nominal, self.ndata)
        mujoco.mj_fullM(self.nominal, self._M, self.ndata.qM)
        return self._M.copy(), self.ndata.qfrc_bias.copy()

    # --- introspection used by the drift wrapper ---
    def body_id(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def joint_dof(self, name):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return self.model.jnt_dofadr[jid]
