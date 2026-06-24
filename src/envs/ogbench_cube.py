"""E4: the real OGBench cube-single benchmark, manipulated under changing physics.

This loads the actual OGBench `cube-single-play-singletask-v0` model (the UR5e workspace, the cube
body, its mass, geometry, and table contact/friction) and manipulates the real cube. We verified
that the real Robotiq pinch grasp cannot carry the cube through a dynamic trajectory (it slips
above ~0.3 kg and drops even the nominal cube during fast motion), so rather than fake a carry we
push the cube across its own table to a goal, which is the contact-rich primitive the friction
drift actually bites on. The arm is held at its home pose, out of the cube's plane, and the cube's
planar position is driven by a force standing in for the arm's net contact push, a lightweight
goal-reaching controller of the kind the build spec explicitly allows.

The nominal model the computed-torque baseline uses is the cube's mass alone, so the real Coulomb
table friction is an unmodeled, nonlinear matched disturbance, and the drift (cube mass up, table
contact friction up) makes that disturbance grow mid-episode. Requires the ogbench package, so this
adapter only imports in the Python 3.11/3.12 `.venv-ogbench` environment.

Adapter interface matches every other environment: nu, reset, q, qd, step(tau), nominal_dynamics,
full_qpos, so it plugs straight into experiments.run_env._rollout_arm.
"""
from __future__ import annotations
import numpy as np
import mujoco
import ogbench

ENV_ID = "cube-single-play-singletask-v0"
CUBE_QADR = 14            # cube free-joint qpos address (x,y,z,quat)
CUBE_DOF = (14, 15)       # planar x,y dof indices we actuate
ORIGIN = np.array([0.4, 0.0])   # cube start position on the table


class OGBenchCube:
    def __init__(self, dt: float = 0.01, substeps: int = 5, nominal_mass: float = 0.5,
                 nominal_friction: float = 0.3):
        env = ogbench.make_env_and_datasets(ENV_ID, env_only=True)
        self.env = env
        self.model = env.unwrapped.model
        self.data = env.unwrapped.data
        self.model.opt.timestep = dt / substeps
        self.dt = dt
        self.substeps = substeps
        self.nu = 2                       # planar push (x, y)
        self.nominal_mass = nominal_mass
        self.nominal_friction = nominal_friction
        self.cube_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object_0")
        self.cube_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_0")
        # Set the nominal mass and operating friction here, before the DriftSchedule snapshots the
        # model, so the drift multipliers scale from these and the baseline tracks well pre-drift.
        self.model.body_mass[self.cube_bid] = nominal_mass
        self.model.geom_friction[self.cube_gid, 0] = nominal_friction
        self._home = None

    def reset(self, q0=None, qd0=None):
        self.env.reset(seed=0)
        d, m = self.data, self.model
        mujoco.mj_forward(m, d)
        self._home = d.qpos[:6].copy()    # hold the arm here, out of the cube plane
        # set the cube on the table at ORIGIN + q0
        start = ORIGIN + (np.asarray(q0) if q0 is not None else 0.0)
        d.qpos[CUBE_QADR:CUBE_QADR + 2] = start
        d.qpos[CUBE_QADR + 2] = 0.03
        d.qpos[CUBE_QADR + 3:CUBE_QADR + 7] = [1, 0, 0, 0]
        d.qvel[CUBE_QADR:CUBE_QADR + 6] = 0
        m.body_mass[self.cube_bid] = self.nominal_mass
        m.geom_friction[self.cube_gid, 0] = self.nominal_friction
        mujoco.mj_forward(m, d)
        # settle the cube onto the table
        for _ in range(40):
            d.qfrc_applied[:] = 0
            d.ctrl[:6] = self._home
            mujoco.mj_step(m, d)
        if qd0 is not None:
            d.qvel[CUBE_DOF[0]], d.qvel[CUBE_DOF[1]] = qd0
        return self.q, self.qd

    @property
    def q(self):
        return self.data.qpos[CUBE_QADR:CUBE_QADR + 2].copy() - ORIGIN

    @property
    def qd(self):
        return self.data.qvel[list(CUBE_DOF)].copy()

    def full_qpos(self):
        return self.data.qpos.copy()

    def step(self, tau):
        d, m = self.data, self.model
        for _ in range(self.substeps):
            d.qfrc_applied[:] = 0
            d.qfrc_applied[CUBE_DOF[0]] = tau[0]
            d.qfrc_applied[CUBE_DOF[1]] = tau[1]
            d.ctrl[:6] = self._home        # keep the arm parked at home, off the table
            mujoco.mj_step(m, d)
        return self.q, self.qd

    def nominal_dynamics(self, q, qd):
        # the baseline knows only the cube's nominal mass; friction is the unmodeled disturbance
        M0 = self.nominal_mass * np.eye(2)
        b0 = np.zeros(2)
        return M0, b0


# --------------------------------------------------------------------------------------------------
ARM_DOF = list(range(6))          # the six UR5e joints
PUSH_Z = 0.04                      # end-effector height for pushing the cube


class OGBenchCubeArm:
    """Arm-in-the-loop variant: the real UR5e end effector pushes the cube to its goal.

    The six UR5e joints are torque-controlled (the built-in position servos are disabled and we drive
    the joints with computed torque through qfrc_applied), the gripper is held closed to act as a flat
    pusher, and the controller tracks a joint reference, produced by online inverse kinematics, that
    advances the end effector forward while staying aligned with the cube so the push stays centered.
    The matched uncertainty is the cube's contact reaction, which grows when the drift raises the cube
    mass and the table friction, so this is a genuine arm-on-object contact task rather than a force
    applied directly to the cube. Tracking error is reported in joint space (deg), the same schema as
    the other arm environments, and the cube-to-goal distance is logged as a task-level metric.
    """

    def __init__(self, dt: float = 0.01, substeps: int = 5, nominal_mass: float = 0.5,
                 nominal_friction: float = 0.5, push_dist: float = 0.26):
        env = ogbench.make_env_and_datasets(ENV_ID, env_only=True)
        self.env = env
        self.model = env.unwrapped.model
        self.data = env.unwrapped.data
        self.ik_data = mujoco.MjData(self.model)
        self.model.opt.timestep = dt / substeps
        self.dt = dt
        self.substeps = substeps
        self.nu = 6
        self.nominal_mass = nominal_mass
        self.nominal_friction = nominal_friction
        self.push_dist = push_dist
        self.episode_T = None
        self.cube_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object_0")
        self.cube_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_0")
        self.tgt_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object_target_0")
        self.pinch = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ur5e/robotiq/pinch")
        # disable the arm position servos so the joints are pure torque inputs
        for i in ARM_DOF:
            self.model.actuator_gainprm[i, :] = 0
            self.model.actuator_biasprm[i, :] = 0
        # nominal mass + operating friction, set before the DriftSchedule snapshots the model
        self.model.body_mass[self.cube_bid] = nominal_mass
        self.model.geom_friction[self.cube_gid, 0] = nominal_friction
        self._M = np.zeros((self.model.nv, self.model.nv))
        self._Jp = np.zeros((3, self.model.nv))
        self._Jr = np.zeros((3, self.model.nv))
        self._ik_warm = None
        self._ee_start_x = None
        self._cube_start_y = None
        self._goal_x = None

    def _ik(self, target, q_init, iters=60, st=0.5, tol=5e-4):
        """Jacobian IK on a scratch MjData so the live sim state is never disturbed."""
        d = self.ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[:6] = q_init
        for _ in range(iters):
            mujoco.mj_forward(self.model, d)
            err = target - d.site_xpos[self.pinch]
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(self.model, d, self._Jp, self._Jr, self.pinch)
            J = self._Jp[:, :6]
            d.qpos[:6] += st * J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
        return d.qpos[:6].copy()

    def reset(self, q0=None, qd0=None):
        d, m = self.data, self.model
        self.env.reset(seed=0)
        # pre-close the gripper so the fingers form a compact pusher
        for _ in range(30):
            d.ctrl[6] = 255
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        # small per-seed perturbation of the cube start (q0 is in metres here)
        dp = np.asarray(q0[:2]) if q0 is not None else np.zeros(2)
        d.qpos[CUBE_QADR:CUBE_QADR + 2] += dp
        m.body_mass[self.cube_bid] = self.nominal_mass
        m.geom_friction[self.cube_gid, 0] = self.nominal_friction
        mujoco.mj_forward(m, d)
        cube = d.qpos[CUBE_QADR:CUBE_QADR + 3].copy()
        self._cube_start_y = float(cube[1])
        self._goal_x = float(d.xpos[self.tgt_bid][0])
        # place the end effector behind the cube with clearance, then settle holding gravity
        ee0 = np.array([cube[0] - 0.12, cube[1], PUSH_Z])
        self._ee_start_x = float(ee0[0])
        q_start = self._ik(ee0, d.qpos[:6].copy())
        self._ik_warm = q_start.copy()
        d.qpos[:6] = q_start
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        for _ in range(40):
            d.ctrl[6] = 255
            mujoco.mj_fullM(m, d, self._M)
            d.qfrc_applied[:6] = d.qfrc_bias[:6]
            mujoco.mj_step(m, d)
        return self.q, self.qd

    def command(self, t):
        """Joint reference from online IK: advance the EE in x while tracking the cube's y."""
        T = self.episode_T or 6.0
        push_x = self._ee_start_x + self.push_dist * min(1.0, t / T)
        cube_y = float(self.data.qpos[CUBE_QADR + 1])
        cube_y = np.clip(cube_y, self._cube_start_y - 0.05, self._cube_start_y + 0.05)
        target = np.array([push_x, cube_y, PUSH_Z])
        self._ik_warm = self._ik(target, self._ik_warm, iters=20)
        return self._ik_warm.copy()

    @property
    def q(self):
        return self.data.qpos[ARM_DOF].copy()

    @property
    def qd(self):
        return self.data.qvel[ARM_DOF].copy()

    def full_qpos(self):
        return self.data.qpos.copy()

    def step(self, tau):
        d, m = self.data, self.model
        for _ in range(self.substeps):
            d.qfrc_applied[:6] = tau
            d.ctrl[6] = 255            # hold the gripper closed as a pusher
            mujoco.mj_step(m, d)
        return self.q, self.qd

    def nominal_dynamics(self, q, qd):
        d, m = self.data, self.model
        mujoco.mj_fullM(m, d, self._M)
        M0 = self._M[:6, :6].copy()
        b0 = d.qfrc_bias[:6].copy()    # arm gravity + Coriolis; the cube contact is the disturbance
        return M0, b0

    def cube_to_goal_cm(self):
        return float(abs(self.data.qpos[CUBE_QADR] - self._goal_x) * 100)
