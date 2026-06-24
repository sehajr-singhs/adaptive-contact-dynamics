"""E3: planar pushing under changing friction, the contact-rich core.

We model the manipuland as a block with two translational (slide) joints on a table, driven by
x/y force actuators that stand in for a pusher's net contact force, with Coulomb joint friction
(`frictionloss`) playing the role of table friction. The nominal twin used for the computed-torque
feedback is frictionless, so the real friction is an unmodeled, nonlinear (stiction / sign-of-
velocity) matched disturbance, which is exactly the contact effect the adaptive term has to cancel.
Drift raises the friction coefficient and the block mass mid-episode, the "surface changed and the
block got heavier" event.
"""
from __future__ import annotations
import numpy as np
import mujoco

XML = """
<mujoco model="push2d">
  <option gravity="0 0 0" integrator="RK4"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight diffuse="0.8 0.8 0.8"/>
  </visual>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="table" type="plane" size="1 1 0.1" pos="0 0 0" rgba="0.93 0.93 0.95 1"/>
    <site name="goal" type="cylinder" size="0.05 0.001" rgba="0.1 0.8 0.2 0.5" pos="0.25 0.2 0.001"/>
    <body name="object" pos="0 0 0.05">
      <joint name="ox" type="slide" axis="1 0 0" frictionloss="{fl}"/>
      <joint name="oy" type="slide" axis="0 1 0" frictionloss="{fl}"/>
      <geom name="obj" type="box" size="0.05 0.05 0.05" mass="1.0" rgba="0.85 0.45 0.2 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="ox" gear="1"/>
    <motor joint="oy" gear="1"/>
  </actuator>
</mujoco>
"""


class Block2D:
    def __init__(self, dt: float = 0.01, substeps: int = 5, frictionloss: float = 1.5):
        self.frictionloss = frictionloss
        self.model = mujoco.MjModel.from_xml_string(XML.format(fl=frictionloss))
        self.nominal = mujoco.MjModel.from_xml_string(XML.format(fl=0.0))  # frictionless twin
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
