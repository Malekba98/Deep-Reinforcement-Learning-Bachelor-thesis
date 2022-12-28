r""""Agent classes for Project Phoenix.

Author:     Sven Gronauer
Created:    14.04.2021
Updates:    17.05.2021
"""
import numpy as np
import pybullet as pb
from pybullet_utils import bullet_client
import abc
import os
import xml.etree.ElementTree as etxml
from typing import Tuple
from scipy import signal
import phoenix_simulation.envs.Control as phoenix_control
from phoenix_simulation.envs.EnvUtils import get_assets_path


class AgentBase(abc.ABC):
    r"""Base class for agents."""
    def __init__(
            self,
            bc: bullet_client.BulletClient,
            control_mode: str,
            name: str,
            file_name: str,
            act_dim: int,
            obs_dim: int,
            frame_skip: int,
            time_step: float,
            aggregate_phy_steps: int,
            init_color: tuple = (1., 1., 1, 1.0),
            init_xyz: tuple = (0., 0., 0.),
            init_orientation: tuple = (0., 0., 0.),
            fixed_base=False,
            global_scaling=1,
            self_collision=False,
            verbose=False,
            debug=False,
            **kwargs
    ):
        assert len(init_orientation) == 3, 'init_orientation expects (r,p,y)'
        assert len(init_xyz) == 3
        self.aggregate_phy_steps = aggregate_phy_steps
        self.bc = bc
        self.name = name
        self.file_name = file_name
        self.fixed_base = 1 if fixed_base else 0
        self.file_name_path = os.path.join(get_assets_path(), self.file_name)
        self.global_scaling = global_scaling
        self.init_xyz = np.array(init_xyz)
        self.init_color = np.array(init_color)
        self.init_orientation = pb.getQuaternionFromEuler(init_orientation)
        self.init_quaternion = pb.getQuaternionFromEuler(init_orientation)
        self.init_rpy = init_orientation
        self.self_collision = self_collision
        self.visible = True
        self.verbose = verbose
        self.frame_skip = frame_skip
        self.time_step = time_step
        self.debug = debug

        # space information
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.body_unique_id = self.load_assets()

        # Setup controller
        assert hasattr(phoenix_control, control_mode), \
            f'Control={control_mode} not found.'
        control_cls = getattr(phoenix_control, control_mode)  # get class reference
        self.control = control_cls(
            self,  # pass Drone class
            self.bc,
            time_step=time_step,  # 1 / sim_frequency
            frame_skip=frame_skip
        )

    @abc.abstractmethod
    def apply_force(self, force):
        """Apply force vector to drone motors."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply_z_torque(self, force):
        """Apply torque responsible for yaw."""
        raise NotImplementedError

    # @abc.abstractmethod
    # def convert_action_to_force(self, action) -> np.ndarray:
    #     """Convert abstract action to force vector [N]."""
    #     raise NotImplementedError

    @abc.abstractmethod
    def get_state(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load_assets(self) -> int:
        """Loads the robot description file into the simulation."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """Agent specific reset."""
        raise NotImplementedError

    def violates_constraints(self, does_violate_constraint):
        """Displays a red sphere which indicates the receiving of costs when
        enable is True, else deactivate visual shape."""
        pass


class CrazyFlieAgent(AgentBase):
    def __init__(
            self,
            bc: bullet_client.BulletClient,
            control_mode: str,
            file_name: str,   # two URDF files are implemented: cf21x_bullet.urdf and cf21x_sys_eq.urdf
            frame_skip: int,
            time_step: float,
            aggregate_phy_steps: int,
            use_motor_dynamics: bool = True
    ):
        super(CrazyFlieAgent, self).__init__(
            bc=bc,
            control_mode=control_mode,
            name='CrazyFlie2.1X',
            file_name=file_name,
            act_dim=4,
            obs_dim=16,
            frame_skip=frame_skip,
            time_step=time_step,
            aggregate_phy_steps=aggregate_phy_steps,
        )
        self._parse_robot_parameters()
        self.use_motor_dynamics = use_motor_dynamics

        # Parameters from Julian Förster:
        # 2.130295e-11 * PWM ** 2 + 1.032633e-6 * PWM + 5.484560e-4
        self.PWM_FORCE_FACTOR_0 = 5.484560e-4
        self.PWM_FORCE_FACTOR_1 = 1.032633e-6
        self.PWM_FORCE_FACTOR_2 = 2.130295e-11
        self.FORCE_TORQUE_FACTOR_0 = 1.56e-5
        self.FORCE_TORQUE_FACTOR_1 = 5.96e-3

        self.m = self.M  # mass of drone in [kg]
        self.xyz = np.zeros(3, dtype=np.float32)  # [m]
        self.quat = np.zeros(4, dtype=np.float32)  # quaternion
        self.rpy = np.zeros(3, dtype=np.float32)  # [rad]
        self.xyz_dot = np.zeros(3, dtype=np.float32)  # [m/s]
        self.rpy_dot = np.zeros(3, dtype=np.float32)  # [rad/s]
        self.last_action = np.zeros(self.act_dim)
        # self.current_motor_speeds = np.zeros(4)  # [rpm]
        # self.y = np.zeros(4)  # [N]

        # From Landry's Master's thesis (p. 39)
        # Note that we also take the delay of the control loop into account when
        # running the optimization by shifting the input tape by the appropriate
        # number of time steps. [...] In our case this resulted in a 28ms delay.
        self.delay = 1  # 10 ms action input delay (default value)
        # self.action_queue = queue.Queue(N)
        # [self.action_queue.put(np.zeros(self.act_dim)) for i in range(N)]
        self.action_buffer = np.zeros(shape=(10, self.act_dim))
        self.action_idx = 0

        # parameters changed by domain randomization
        self.kf = self.KF
        self.km = self.KM
        self.pwm_force_factor_0 = self.PWM_FORCE_FACTOR_0
        self.pwm_force_factor_1 = self.PWM_FORCE_FACTOR_1
        self.pwm_force_factor_2 = self.PWM_FORCE_FACTOR_2
        self.force_torque_factor_0 = self.FORCE_TORQUE_FACTOR_0
        self.force_torque_factor_1 = self.FORCE_TORQUE_FACTOR_1

        # === setup motor dynamics (PT1 behavior)
        self.K = 7.2345374e-8
        self.T_s_T = 0.9695404
        # self.k = self.K  # set by domain randomization
        # self.t_s_t = self.T_s_T  # set by domain randomization
        dt = self.time_step * self.frame_skip
        if self.use_motor_dynamics:
            assert dt == 1 / 500, \
                f'Not implemented for frequency other than 500Hz. Got: {1/dt}'
        # Note that A, B, C, D are manipulated by domain randomization
        self.A, self.B, self.C, self.D = self._build_system()

        self.init_force = 0.08  # [N] per motor
        self.x = self.init_force * np.ones(4) / self.C  # since x = y / C

    def _build_system(self) -> tuple:
        """Build discrete system description of motor behavior."""
        num = [0, self.K]
        den = [1, -self.T_s_T]
        # Time delta with which drone.apply_action() is called
        dt = self.time_step * self.frame_skip

        # Transfer function for motor forces
        # x(k+1) = A x(k) + B u(k)
        # y(k) = C x(k)
        tf = signal.TransferFunction(num, den, dt=dt)
        sys = tf.to_ss()
        A = np.ones(self.act_dim) * float(sys.A)
        B = np.ones(self.act_dim) * float(sys.B)
        C = np.ones(self.act_dim) * float(sys.C)
        D = np.ones(self.act_dim) * float(sys.D)
        return A, B, C, D

    def _parse_robot_parameters(self) -> None:
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.
        """
        URDF_TREE = etxml.parse(self.file_name_path).getroot()
        self.M = float(URDF_TREE[1][0][1].attrib['value'])
        self.L = float(URDF_TREE[0].attrib['arm'])
        self.THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        self.IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        self.IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        self.IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        self.J = np.diag([self.IXX, self.IYY, self.IZZ])
        self.J_INV = np.linalg.inv(self.J)
        self.KF = float(URDF_TREE[0].attrib['kf'])
        self.KM = float(URDF_TREE[0].attrib['km'])
        self.COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        self.COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        self.COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        self.COLLISION_Z_OFFSET = self.COLLISION_SHAPE_OFFSETS[2]
        self.MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        self.GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        self.PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        self.DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])  # [kg /rad]
        self.DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])  # [kg /rad]
        self.DRAG_COEFF = np.array([self.DRAG_COEFF_XY, self.DRAG_COEFF_XY, self.DRAG_COEFF_Z])  # [kg /rad]
        self.DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        self.DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        self.DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])

        # print(f'===============')
        # print(f'KF: {self.KF}')
        # print(f'KM: {self.KM}')
        # print(f'===============')

    def apply_action(self, action) -> Tuple[np.ndarray, float]:
        """Returns the forces that are applied to drone motors."""

        # ====== Note: disabled delayed actions
        # get delayed action first
        # delayed_action = self.action_buffer[self.action_idx].copy()
        # then set current action for later
        # self.action_buffer[self.action_idx] = action
        # self.action_idx = (self.action_idx + 1) % self.delay

        # action to PWM signal based on the control mode (PWM, Attitude Rate)
        PWMs = self.control.act(action=action)

        if self.use_motor_dynamics:
            # Update motor speeds (LTI model)
            self.x = self.A * self.x + self.B * PWMs  # x(k+1) = A x(k) + B u(k)
            current_motor_forces = self.y  # y(k+1) = C x(k+1)
        else:
            # Equation from BA Thesis of Julian Foerster
            current_motor_forces = self.pwm_force_factor_2 * PWMs**2 \
                                   + self.pwm_force_factor_1 * PWMs \
                                   + self.pwm_force_factor_0
        torques = self.force_torque_factor_1 * current_motor_forces \
                  + self.force_torque_factor_0
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])

        return current_motor_forces, z_torque

    def apply_force(self, force, frame=pb.LINK_FRAME):
        """Apply a force vector to mass center of drone."""
        assert force.size == 3
        self.bc.applyExternalForce(
            self.body_unique_id,
            4,  # center of mass link
            forceObj=force,
            posObj=[0, 0, 0],
            flags=frame
        )

    def apply_motor_forces(self, forces):
        """Apply a force vector to the drone motors."""
        assert forces.size == 4
        for i in range(4):
            self.bc.applyExternalForce(
                self.body_unique_id,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=pb.LINK_FRAME
            )

    def apply_z_torque(self, torque):
        """Apply torque responsible for yaw."""
        self.bc.applyExternalTorque(
            self.body_unique_id,
            4,  # center of mass link
            torqueObj=[0, 0, torque],
            flags=pb.LINK_FRAME
        )

    @property
    def y(self) -> np.ndarray:
        """Forces that currently act on drone rotors (in [N])."""
        return self.x * self.C  # Linear state-space model: y(k) = C x(k)

    def get_state(self):
        state = np.hstack([self.xyz,
                           self.quat,
                           self.rpy,
                           self.xyz_dot,
                           self.rpy_dot,
                           self.last_action
                           ])
        return state.reshape(20, )

    def load_assets(self) -> int:
        """Loads the robot description file into the simulation.

        Expected file format: URDF

        Returns
        -------
            body_unique_id of loaded body
        """
        assert self.file_name_path.endswith('.urdf')
        assert os.path.exists(self.file_name_path), \
            f'Did not find {self.file_name} at: {get_assets_path()}'
        # print(f'file_name_path: {self.file_name_path} ')
        random_xyz = (0, 0, 0)
        random_rpy = (0, 0, 0)

        body_unique_id = self.bc.loadURDF(
            self.file_name_path,
            random_xyz,
            pb.getQuaternionFromEuler(random_rpy),
            # Important Note: take inertia from URDF...
            flags=pb.URDF_USE_INERTIA_FROM_FILE
        )
        assert body_unique_id >= 0  # , msg
        return body_unique_id

    def reset(self) -> None:
        """Agent specific reset function."""
        self.last_action = np.zeros(self.act_dim)
        self.control.reset()  # reset PID control or PWM control
        self.x = self.init_force * np.ones(4) / self.C

    def update_information(self):
        """"Retrieve drone's kinematic information from PyBullet simulation."""
        pos, quat = self.bc.getBasePositionAndOrientation(self.body_unique_id)
        self.xyz = np.array(pos)
        self.quat = np.array(quat)
        self.rpy = np.array(self.bc.getEulerFromQuaternion(quat))
        xyz_dot, rpy_dot = self.bc.getBaseVelocity(self.body_unique_id)
        self.xyz_dot = np.array(xyz_dot)
        self.rpy_dot = np.array(rpy_dot)


class CrazyFlieBulletAgent(CrazyFlieAgent):
    def __init__(
            self,
            bc: bullet_client.BulletClient,
            control_mode: str,
            frame_skip: int,
            time_step: float,
            aggregate_phy_steps: int,
    ):
        super().__init__(
            bc=bc,
            control_mode=control_mode,
            file_name='cf21x_bullet.urdf',
            frame_skip=frame_skip,
            time_step=time_step,
            aggregate_phy_steps=aggregate_phy_steps,
            use_motor_dynamics=True  # model PT-1 motor dynamics
        )
        
        
class CrazyFlieSysEqAgent(CrazyFlieAgent):
    def __init__(
            self,
            bc: bullet_client.BulletClient,
            control_mode: str,
            frame_skip: int,
            time_step: float,
            aggregate_phy_steps: int,
    ):
        super(CrazyFlieSysEqAgent, self).__init__(
            bc=bc,
            control_mode=control_mode,
            file_name='cf21x_sys_eq.urdf',  # use
            frame_skip=frame_skip,
            time_step=time_step,
            aggregate_phy_steps=aggregate_phy_steps,
            use_motor_dynamics=False  # disable PT-1 motor dynamics
        )
