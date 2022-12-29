"""
Copyright (c) 2022 Sven Gronauer (Technical University of Munich)

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
import pybullet as pb
import pybullet_data
import gym
from pybullet_utils import bullet_client
import abc
import os
import phoenix_simulation.envs.Physics as phoenix_physics
from phoenix_simulation.envs.Agents import CrazyFlieSysEqAgent, \
    CrazyFlieBulletAgent
from phoenix_simulation.envs.EnvUtils import get_assets_path

#
# def get_assets_path() -> str:
#     r""" Returns the path to the files located in envs/data."""
#     data_path = os.path.join(os.path.dirname(__file__), 'assets')
#     return data_path


class DroneBaseEnv(gym.Env, abc.ABC):
    """Base class for all drone environments."""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            physics: str,  # e.g. PyBulletPhysics or SimplifiedSystemEquations
            control_mode: str,  # e.g. pwm, attitude_rate
            drone_model: str,  # options: [cf21x_bullet, cf21x_sys_eq]
            initial_xyzs=None,
            initial_rpys=None,
            initial_xyz_dots=None,
            initial_rpy_dots=None,
            sim_freq: int = 200,
            aggregate_phy_steps: int = 5,
            frame_skip: int = 2,  # use 100Hz control for drones
            domain_randomization: float = -1,  # deactivated when negative value
            observation_noise=0.0,  # default: no noise added to obs
            action_noise=0.0,  # default: no noise added to actions
            enable_reset_distribution=True,
            graphics=False,
            debug=False
    ):
        """

        ::Notes::
        - Domain Randomization (DR) is applied when calling reset method and the
            domain_randomization is a positive float.

        Parameters
        ----------
        physics: str
            Name of physics class to be instantiated.
        drone_model
        initial_xyzs
        initial_rpys
        initial_xyz_dots
        initial_rpy_dots
        sim_freq
        aggregate_phy_steps
        frame_skip
        domain_randomization:
            Apply domain randomization to system parameters if value > 0
        graphics
        debug

        Raises
        ------
        AssertionError
            If no class is found for given physics string.
        """
        self.input_parameters = locals()  # save setting for later reset
        self.use_graphics = graphics
        self.domain_randomization = domain_randomization
        self.drone_model = drone_model
        self.debug = debug
        self.enable_reset_distribution = enable_reset_distribution

        # Default simulation constants (in capital letters)
        self.G = 9.8
        self.RAD_TO_DEG = 180 / np.pi
        self.DEG_TO_RAD = np.pi / 180
        self.SIM_FREQ = sim_freq  # default: 200Hz
        self.TIME_STEP = 1. / self.SIM_FREQ  # default: 0.005

        # Physics parameters depend on the task
        self.time_step = self.TIME_STEP
        self.frame_skip = frame_skip  # default: 2 -> 100 Hz control
        self.number_solver_iterations = 5
        self.aggregate_phy_steps = aggregate_phy_steps

        # === Initialize and setup PyBullet ===
        self.bc = self._setup_client_and_physics(self.use_graphics)
        self.stored_state_id = -1

        # === spawn plane and drone agent ===
        self._setup_simulation(physics=physics, control_mode=control_mode)

        # === Observation space and action space ===
        # negative noise values denote that zero noise is applied
        self.action_noise = action_noise
        self.observation_noise = observation_noise
        obs_dim = self.compute_observation().size
        act_dim = self.drone.act_dim
        # Define limits for observation space and action space
        o_lim = 1000 * np.ones((obs_dim, ), dtype=np.float32)
        a_lim = np.ones((act_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-o_lim, o_lim, dtype=np.float32)
        self.action_space = gym.spaces.Box(-a_lim, a_lim, dtype=np.float32)

        # stepping information
        self.iteration = 0
        self.old_potential = self.compute_potential()

        # task specific parameters (may be adjusted by child classes)
        self.init_pos = np.array([0, 0, 1], dtype=np.float32)
        self.init_rpy = np.zeros(3)
        self.init_xyz_dot = np.zeros(3)
        self.init_rpy_dot = np.zeros(3)

    def _setup_client_and_physics(
            self,
            graphics=False
    ) -> bullet_client.BulletClient:
        r"""Creates a PyBullet process instance.

        The parameters for the physics simulation are determined by the
        get_physics_parameters() function.

        Parameters
        ----------
        graphics: bool
            If True PyBullet shows graphical user interface with 3D OpenGL
            rendering.

        Returns
        -------
        bc: BulletClient
            The instance of the created PyBullet client process.
        """
        if graphics or self.use_graphics:
            bc = bullet_client.BulletClient(connection_mode=pb.GUI)
        else:
            bc = bullet_client.BulletClient(connection_mode=pb.DIRECT)

        # add open_safety_gym/envs/data to the PyBullet data path
        bc.setAdditionalSearchPath(get_assets_path())
        # disable GUI debug visuals
        bc.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        bc.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
        bc.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step * self.frame_skip,
            numSolverIterations=self.number_solver_iterations,
            deterministicOverlappingPairs=1,
            numSubSteps=self.frame_skip)
        bc.setGravity(0, 0, -9.81)
        # bc.setDefaultContactERP(0.9)
        return bc

    def _setup_simulation(self, physics, control_mode) -> None:
        r"""Create world layout, spawn agent and obstacles.

        Takes the passed parameters from the class instantiation: __init__().
        """
        # reset some variables that might be changed by DR -- this avoids errors
        # when calling the render() method after training.
        self.g = self.G
        self.time_step = self.TIME_STEP

        # also add PyBullet's data path
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.PLANE_ID = self.bc.loadURDF("plane.urdf")
        # Load 20x20 Walls
        # pb.loadURDF(assets_path + "room_20x20.urdf", useFixedBase=True)
        # random spawns

        if self.drone_model == 'cf21x_bullet':
            self.drone = CrazyFlieBulletAgent(
                bc=self.bc, control_mode=control_mode,
                time_step=self.time_step,  
                frame_skip=self.frame_skip,
                aggregate_phy_steps=self.aggregate_phy_steps
            )
        elif self.drone_model == 'cf21x_sys_eq':
            self.drone = CrazyFlieSysEqAgent(
                bc=self.bc, control_mode=control_mode,
                time_step=self.time_step,
                frame_skip=self.frame_skip,
                aggregate_phy_steps=self.aggregate_phy_steps
            )
        else:
            raise NotImplementedError

        # Setup forward dynamics - Instantiates a particular physics class.
        assert hasattr(phoenix_physics, physics), \
            f'Physics={physics} not found.'
        physics_cls = getattr(phoenix_physics, physics)  # get class reference
        # call class constructor
        self.physics = physics_cls(
            self.drone,
            self.bc,
            time_step=self.time_step,  # 1 / sim_frequency
            frame_skip=self.frame_skip,
        )

        # Setup task specifics
        self._setup_task_specifics()

    @abc.abstractmethod
    def _setup_task_specifics(self):
        raise NotImplementedError

    def apply_domain_randomization(self) -> None:
        """ Apply domain randomization at the start of every new episode.

            Parameters:
                - M: mass of drone
                - KM:
                - KF:
                - time_step
        """
        # Initialize simulation constants used for domain randomization:
        # the following values are reset at the beginning of every epoch

        def drawn_new_value(default_value,
                            factor=self.domain_randomization,
                            size=None):
            """Draw a random value from a uniform distribution."""
            bound = factor * default_value
            bounds = (default_value - bound, default_value + bound)
            return np.random.uniform(*bounds, size=size)

        if self.domain_randomization > 0:
            # physics parameter
            self.time_step = drawn_new_value(self.TIME_STEP)
            self.physics.set_parameters(
                time_step=self.time_step,
                frame_skip=self.frame_skip,
                number_solver_iterations=self.number_solver_iterations,
            )

            # === Drone parameters ====
            self.drone.m = drawn_new_value(self.drone.M)
            J_diag = np.array([self.drone.IXX, self.drone.IYY, self.drone.IZZ])
            J_diag_sampled = drawn_new_value(J_diag, size=3)
            self.drone.J = np.diag(J_diag_sampled)
            self.drone.J_INV = np.linalg.inv(self.drone.J)

            self.drone.force_torque_factor_0 = drawn_new_value(
                self.drone.FORCE_TORQUE_FACTOR_0)
            self.drone.force_torque_factor_1 = drawn_new_value(
                self.drone.FORCE_TORQUE_FACTOR_1)
            # add dead times: actuators, communication delays
            self.drone.delay = np.random.randint(low=1, high=4)  # interval [10, 30] ms
            self.drone.action_idx = 0
            self.drone.action_buffer = np.zeros_like(self.drone.action_buffer)

            if self.drone.use_motor_dynamics:
                # set internal motor state variable (current motor force y)
                self.drone.x = drawn_new_value(0.08 / self.drone.C, size=4)
                # Note:: Use DR of 0.5% for motor dynamics time constant
                #   Any DR factor > 0.01 made learning difficult / impossible
                #   Any DR factor > 0.03 causes non-physical values: C > 1
                self.drone.A = drawn_new_value(
                    self.drone.T_s_T, factor=0.005, size=4)
                self.drone.C = drawn_new_value(self.drone.K, size=4)
            else:
                # PWM parameters: 2.13e-11 * PWM ** 2 + 1.03e-6 * PWM + 5.48e-4
                self.drone.pwm_force_factor_0 = drawn_new_value(
                    self.drone.PWM_FORCE_FACTOR_0, size=4)
                self.drone.pwm_force_factor_1 = drawn_new_value(
                    self.drone.PWM_FORCE_FACTOR_1, size=4)
                self.drone.pwm_force_factor_2 = drawn_new_value(
                    self.drone.PWM_FORCE_FACTOR_2, size=4)

            # set new mass and inertia to PyBullet
            self.bc.changeDynamics(
                bodyUniqueId=self.drone.body_unique_id,
                linkIndex=-1,
                mass=self.drone.m,
                localInertiaDiagonal=J_diag_sampled
            )
        else:
            pass

    @abc.abstractmethod
    def compute_done(self) -> bool:
        """Implemented by child classes."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_info(self) -> dict:
        """Implemented by child classes."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_observation(self) -> np.ndarray:
        """Returns the current observation of the environment."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_potential(self) -> float:
        """Implemented by child classes."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_reward(self, action) -> float:
        """Implemented by child classes."""
        raise NotImplementedError

    def render(
            self,
            mode='human'
    ) -> np.ndarray:
        """Show PyBullet GUI visualization.

        Render function triggers the PyBullet GUI visualization.
        Camera settings are managed by Task class.

        Note: For successful rendering call env.render() before env.reset()

        Parameters
        ----------
        mode: str

        Returns
        -------
        array
            holding RBG image of environment if mode == 'rgb_array'
        """
        if mode == 'human':
            # close direct connection to physics server and
            # create new instance of physics with GUI visuals
            if not self.use_graphics:
                self.bc.disconnect()
                self.use_graphics = True
                self.bc = self._setup_client_and_physics(graphics=True)
                self._setup_simulation(self.input_parameters['physics'],
                                       self.input_parameters['control_mode'])
                # Save the current PyBullet instance as save state
                # => This avoids errors when enabling rendering after training
                self.stored_state_id = self.bc.saveState()
        if mode != "rgb_array":
            return np.array([])
        else:
            view_matrix = self.bc.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.world.camera.cam_base_pos,
                distance=self.world.camera.cam_dist,
                yaw=self.world.camera.cam_yaw,
                pitch=self.world.camera.cam_pitch,
                roll=0,
                upAxisIndex=2
            )
            w = float(self.world.camera.render_witime_steph)
            h = self.world.camera.render_height
            proj_matrix = self.bc.computeProjectionMatrixFOV(
                fov=60,
                aspect=w / h,
                nearVal=0.1,
                farVal=100.0
            )
            (_, _, px, _, _) = self.bc.getCameraImage(
                witime_steph=self.world.camera.render_witime_steph,
                height=self.world.camera.render_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=pb.ER_BULLET_HARDWARE_OPENGL)

            new_shape = (self.world.camera.render_height,
                         self.world.camera.render_witime_steph,
                         -1)
            rgb_array = np.reshape(np.array(px), new_shape)
            rgb_array = rgb_array[:, :, :3]
            return rgb_array

    def reset(self) -> np.ndarray:
        """Reset environment to initial state.

        This function is called after agent encountered terminal state.

        Returns
        -------
        array
            holding the observation of the initial state
        """
        # disable rendering before resetting
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 0)
        if self.stored_state_id >= 0:
            self.bc.restoreState(self.stored_state_id)
        else:
            # Restoring a saved state circumvents the necessity to load all
            # bodies again..
            self.stored_state_id = self.bc.saveState()
        self.iteration = 0
        self.task_specific_reset()
        self.drone.reset()  # reset drone must be reset after domain random.
        self.apply_domain_randomization()
        self.bc.stepSimulation()
        # collect information from PyBullet simulation
        """Gather information from PyBullet about drone's current state."""
        self.drone.update_information()
        self.old_potential = self.compute_potential()
        if self.use_graphics:  # enable rendering again after resetting
            self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 1)
        return self.compute_observation()

    def step(
            self,
            action: np.ndarray
    ) -> tuple:
        """Step the simulation's dynamics once forward.

        This method follows the interface of the OpenAI Gym.

        Parameters
        ----------
        action: array
            Holding the control commands for the agent.

        Returns
        -------
        observation (object)
            Agent's observation of the current environment
        reward (float)
            Amount of reward returned after previous action
        done (bool)
            Whether the episode has ended, handled by the time wrapper
        info (dict)
            contains auxiliary diagnostic information such as the cost signal
        """
        original_action = action.copy()
        # action = np.squeeze(action)
        self.iteration += 1
        # Apply action to agent, step forward and collect information
        if self.action_noise > 0:
            action += np.random.uniform(
                -self.action_noise,
                self.action_noise,
                size=action.size)

        for _ in range(self.aggregate_phy_steps):
            self.physics.step_forward(action)

        r = self.compute_reward(action)
        info = self.compute_info()
        done = self.compute_done()
        next_obs = self.compute_observation()
        self.drone.last_action = original_action
        return next_obs, r, done, info

    @abc.abstractmethod
    def task_specific_reset(self):
        """Inheriting child classes define reset environment reset behavior."""
        raise NotImplementedError
