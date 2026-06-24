"""E4: a 3-link arm carrying a cube through a pick-and-place trajectory under changing payload.

This is a self-contained MuJoCo manipulation task standing in for OGBench cube-single. OGBench
itself does not install on Python 3.14 (its dm_control / labmaze dependency has no wheel and will
not build), so rather than fake that benchmark we model the same physical situation: an arm that
has grasped a cube and carries it from a pick pose to a place pose, where the cube turns out to be
heavier than the nominal model assumed. The cube's weight is a gravity-dependent matched
disturbance, and the drift (cube mass up, joint damping up) fires between pick and place, the
"picked it up and it's heavier than expected" event. The arm is gravity-loaded, so a fixed
computed-torque controller sags under the heavier payload and the adaptive term has to cancel it.
"""
from __future__ import annotations
import numpy as np
import mujoco

XML = """
<mujoco model="carry3link">
  <option gravity="0 0 -9.81" integrator="RK4"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight diffuse="0.7 0.7 0.7"/>
  </visual>
  <worldbody>
    <light pos="0 -1 1.5" dir="0 1 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 -0.9" rgba="0.92 0.92 0.94 1"/>
    <site name="place" type="sphere" size="0.03" rgba="0.1 0.8 0.2 0.5" pos="0.5 0 -0.3"/>
    <body name="link1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.15"/>
      <geom name="g1" type="capsule" fromto="0 0 0 0.25 0 0" size="0.028" mass="1.0" rgba="0.20 0.45 0.85 1"/>
      <body name="link2" pos="0.25 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.15"/>
        <geom name="g2" type="capsule" fromto="0 0 0 0.25 0 0" size="0.024" mass="0.7" rgba="0.30 0.55 0.9 1"/>
        <body name="link3" pos="0.25 0 0">
          <joint name="j3" type="hinge" axis="0 1 0" damping="0.15"/>
          <geom name="g3" type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.4" rgba="0.4 0.6 0.95 1"/>
          <body name="cube" pos="0.2 0 0">
            <geom name="cubeg" type="box" size="0.035 0.035 0.035" mass="0.3" rgba="0.85 0.45 0.2 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" gear="1"/>
    <motor joint="j2" gear="1"/>
    <motor joint="j3" gear="1"/>
  </actuator>
</mujoco>
"""


class CarryArm:
    def __init__(self, dt: float = 0.01, substeps: int = 5):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.nominal = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.ndata = mujoco.MjData(self.nominal)
        self.model.opt.timestep = dt / substeps
        self.nominal.opt.timestep = dt / substeps
        self.dt = dt
        self.substeps = substeps
        self.nu = self.model.nu
        self.nq = self.model.nq
        self._M = np.zeros((self.model.nv, self.model.nv))

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
        self.ndata.qpos[:] = q
        self.ndata.qvel[:] = qd
        mujoco.mj_forward(self.nominal, self.ndata)
        mujoco.mj_fullM(self.nominal, self._M, self.ndata.qM)
        return self._M.copy(), self.ndata.qfrc_bias.copy()
