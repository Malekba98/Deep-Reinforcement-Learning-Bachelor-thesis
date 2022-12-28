import numpy as np

from phoenix_simulation.envs.DroneBase import DroneBaseEnv


DEG_TO_RAD = np.pi / 180


class DroneHoverBaseEnv(DroneBaseEnv):
    def __init__(
            self,
            physics,
            control_mode: str,
            drone_model: str,
            action_noise: float = 0.01,
            observation_noise=1,  # must be positive in order to add noise
            domain_randomization=0.10,  # default DR: 10%
            target_pos: np.ndarray = np.array([0, 0, 1], dtype=np.float32),
            sim_freq=500,
            frame_skip=1,  # use 100Hz control for drones
            aggregate_phy_steps=5,  # PID Controller needs to aggregate steps to improve numerical stability
            **kwargs
    ):
        # must be defined before calling super class constructor:
        self.circle_radius = 0.5
        self.target_pos = target_pos  # used in _computePotential()

        # The following constants are used for cost calculation:
        self.vel_limit = 0.25  # [m/s]
        self.roll_pitch_limit = 10 * DEG_TO_RAD  # [rad]
        self.rpy_dot_limit = 100 * DEG_TO_RAD  # [rad/s]
        self.x_lim = 0.15
        self.y_lim = 0.15
        self.z_lim = 1.20

        super(DroneHoverBaseEnv, self).__init__(
            control_mode=control_mode,
            drone_model=drone_model,
            physics=physics,
            action_noise=action_noise,
            observation_noise=observation_noise,
            domain_randomization=domain_randomization,
            sim_freq=sim_freq,
            frame_skip=frame_skip,  # use 100Hz control for drones
            aggregate_phy_steps=aggregate_phy_steps,
            **kwargs
        )

    def _setup_task_specifics(self):
        """Initialize task specifics. Called by _setup_simulation()."""
        # print(f'Spawn target pos at:', self.target_pos)
        target_visual = self.bc.createVisualShape(
            self.bc.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[0.95, 0.1, 0.05, 0.4],
        )
        # Spawn visual without collision shape
        self.bc.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=target_visual,
            basePosition=self.target_pos
        )

    def compute_done(self) -> bool:
        """ Note: the class is wrapped by Gym's Time-wrapper, which returns
        done=True when T >= time_limit."""
        roll, pitch = self.drone.rpy[:2]  # [rad]
        d = 30 * self.DEG_TO_RAD
        z = self.drone.xyz[2]
        return False if abs(pitch) <= d and abs(roll) <= d and z >= .2 else True

    def compute_info(self) -> dict:
        state = self.drone.get_state()
        c = 0.
        info = {}
        # xyz bounds
        x, y, z = state[:3]
        if np.abs(x) > self.x_lim or np.abs(y) > self.y_lim or z > self.z_lim:
            c = 1.
            info['xyz_limit'] = state[:3]
        # roll pitch bounds
        rpy = self.drone.rpy
        if (np.abs(rpy[:2]) > self.roll_pitch_limit).any():
            c = 1.
            info['rpy'] = rpy
        # linear velocities
        if (np.abs(state[10:13]) > self.vel_limit).any():
            c = 1.
            info['xzy_dot'] = state[10:13]
        # angular velocities
        if (np.abs(state[13:16]) > self.rpy_dot_limit).any():
            c = 1.
            info['rpy_dot'] = state[13:16] * 180 / np.pi
        # update ron visuals when costs are received
        # self.violates_constraints(True if c > 0 else False)

        info['cost'] = c
        return info

    def compute_observation(self) -> np.ndarray:
        state = self.drone.get_state()
        obs = np.hstack([
            state[0:3],  # xyz
            # state[3:7],  # quaternion
            state[7:10],  # roll, pitch, yaw
            state[10:13],  # xyz_dot
            state[13:16],  # rpy_dot
            state[16:20]  # last_Action
            ]).reshape(16, )

        if self.observation_noise > 0:  # add noise only for positive values
            xyz_noise = 0.002
            obs[0:3] += np.random.normal(0, xyz_noise, size=3)
            rpy_noise = 0.1 * self.DEG_TO_RAD
            obs[3:6] += np.random.normal(0, rpy_noise, size=3)
            xyz_dot_noise = 0.02
            obs[6:9] += np.random.uniform(0, xyz_dot_noise, size=3)
            rpy_dot_noise = 1 * self.DEG_TO_RAD
            obs[9:12] += np.random.uniform(0, rpy_dot_noise, size=3)
        return obs

    def compute_potential(self) -> float:
        """Euclidean distance from current ron position to target position."""
        dist = np.linalg.norm(self.drone.xyz - self.target_pos)
        return dist

    def compute_reward(self, action) -> float:
        # Determine penalties
        spin_penalty = 1e-4 * np.linalg.norm(self.drone.rpy_dot)**2
        terminal_penalty = 10 if self.compute_done() else 0.
        action_penalty = 5e-3 * np.linalg.norm(action)**2
        velocity_penalty = 0.01 * np.linalg.norm(self.drone.xyz_dot)**2
        act_diff = action - self.drone.last_action
        ARP = 1.0e-3 * np.linalg.norm(act_diff)**2   # action_rate_penalty
        penalties = spin_penalty + terminal_penalty + action_penalty \
                    + ARP + velocity_penalty
        # Reward shaping with potential
        current_potential = self.compute_potential()
        r = self.old_potential - current_potential
        self.old_potential = current_potential
        reward = 100 * r - penalties
        return reward

    def task_specific_reset(self):
        # set random offset for position
        pos = self.init_pos
        # print('init_pos:', pos )
        rpy = self.init_rpy
        xyz_dot = self.init_xyz_dot
        rpy_dot = self.init_rpy_dot

        if self.enable_reset_distribution:
            pos_lim = 0.02   # default: 0.02
            # Note: use pos + noise instead pos += noise to avoid call by ref
            pos = pos + np.random.uniform(-pos_lim, pos_lim, size=3)

            rpy_lim = 5 * self.DEG_TO_RAD  # default: 5
            rpy = rpy + np.random.uniform(-rpy_lim, rpy_lim, size=3)

            # set random initial velocities
            vel_lim = 0.05  # default: 0.05
            xyz_dot = xyz_dot + np.random.uniform(-vel_lim, vel_lim, size=3)
            rpy_dot_lim = 10 * self.DEG_TO_RAD  # default: 10
            rpy_dot = rpy_dot + np.random.uniform(-rpy_dot_lim, rpy_dot_lim, size=3)
        # print('pos:', pos)
        self.bc.resetBasePositionAndOrientation(
            self.drone.body_unique_id,
            pos,
            self.bc.getQuaternionFromEuler(rpy)
        )
        self.bc.resetBaseVelocity(
            self.drone.body_unique_id,
            linearVelocity=xyz_dot,
            angularVelocity=rpy_dot
        )


""" ==================
    PWM control
"""


class DroneHoverPWMSysEqEnv(DroneHoverBaseEnv):
    def __init__(self):
        super(DroneHoverPWMSysEqEnv, self).__init__(
            control_mode='PWM',
            drone_model='cf21x_sys_eq',
            physics='SystemEquations',
            # use 100 Hz since no motor dynamics and PID is used
            sim_freq=100,
            frame_skip=1,
            aggregate_phy_steps=1,
        )


class DroneHoverPWMBulletEnv(DroneHoverBaseEnv):
    def __init__(self):
        super(DroneHoverPWMBulletEnv, self).__init__(
            control_mode='PWM',
            drone_model='cf21x_bullet',
            physics='PyBulletPhysics',
            sim_freq=500,
            frame_skip=1,
            aggregate_phy_steps=5,  # sub-steps used to calculate motor dynamics
        )


""" ==================
    Attitude Rate control
"""


class DroneHoverPIDSysEqEnv(DroneHoverBaseEnv):
    def __init__(self):
        super(DroneHoverPIDSysEqEnv, self).__init__(
            control_mode='AttitudeRate',
            drone_model='cf21x_sys_eq',
            physics='SystemEquations',
            # === Note: Use 100Hz NN control, but simulate physics and PID with
            # 500Hz to improve numerical stability
            sim_freq=500,
            frame_skip=1,
            aggregate_phy_steps=5,  # number of PID control sub-steps
        )


class DroneHoverPIDBulletEnv(DroneHoverBaseEnv):
    def __init__(self):
        super(DroneHoverPIDBulletEnv, self).__init__(
            control_mode='AttitudeRate',
            drone_model='cf21x_bullet',
            physics='PyBulletPhysics',
            # === Note: Use 100Hz NN control, but simulate physics and PID with
            # 500Hz to improve numerical stability
            sim_freq=500,
            frame_skip=1,
            aggregate_phy_steps=5,  # number of PID control sub-steps
        )
