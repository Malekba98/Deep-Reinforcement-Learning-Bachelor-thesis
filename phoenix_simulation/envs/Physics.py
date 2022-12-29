"""
Copyright (c) 2022 Sven Gronauer (Technical University of Munich)

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import abc
import numpy as np
import pybullet as pb
from pybullet_utils import bullet_client


class BasePhysics(abc.ABC):
    """Parent class."""

    def __init__(
            self,
            drone,
            bc: bullet_client.BulletClient,
            time_step,
            frame_skip,
            gravity: float = 9.81,
            number_solver_iterations: int = 5,
    ):
        self.drone = drone
        self.bc = bc
        self.time_step = time_step
        self.frame_skip = frame_skip
        self.g = gravity
        self.number_solver_iterations = number_solver_iterations

    def set_parameters(
            self,
            time_step,
            frame_skip,
            number_solver_iterations,
            # **kwargs
    ):
        self.time_step = time_step
        self.frame_skip = frame_skip
        self.number_solver_iterations = number_solver_iterations

        # print('new ts:', time_step)
    
    @abc.abstractmethod
    def step_forward(self, action, *args, **kwargs):
        pass


class PyBulletPhysics(BasePhysics):
    
    def set_parameters(self, **kwargs):
        super(PyBulletPhysics, self).set_parameters(**kwargs)
        # Update PyBullet Physics
        self.bc.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step,
            numSolverIterations=self.number_solver_iterations,
            deterministicOverlappingPairs=1,
            numSubSteps=self.frame_skip
        )

    def step_forward(self, action, *args, **kwargs):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        action
        """
        # calculate current motor forces (incorporates delays with motor speeds)
        motor_forces, z_torque = self.drone.apply_action(action)

        # Set motor forces (thrust) and yaw torque in PyBullet simulation
        self.drone.apply_motor_forces(motor_forces)
        self.drone.apply_z_torque(z_torque)

        # === add drag effect
        quat = self.drone.quat
        vel = self.drone.xyz_dot
        base_rot = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)

        rpm = np.sqrt(motor_forces / self.drone.KF)  # force -> motor velocity
        # Simple draft model applied to the base/center of mass
        drag_factors = -1 * self.drone.DRAG_COEFF * np.sum(np.array(2*np.pi*rpm/60))
        drag = np.dot(base_rot, drag_factors*np.array(vel))
        self.drone.apply_force(force=drag)

        # step simulation once forward and collect information from PyBullet
        self.bc.stepSimulation()
        self.drone.update_information()


class SystemEquations(BasePhysics):
    """Simplified version of system difference equations."""

    def step_forward(
            self,
            action,
            *args,
            **kwargs
    ):
        """Explicit but simplified model dynamics implementation.
        Exclude Coriolis term in dynamics.

        Parameters
        ----------
        action:
            action computed by agent policy
        """
        # calculate current motor forces (incorporates delays with motor speeds)
        forces, z_torque = self.drone.apply_action(action)
        thrust = np.array([0, 0, np.sum(forces)])

        # Retrieve current state from agent: (copy to avoid call by reference)
        pos = self.drone.xyz.copy()
        quat = self.drone.quat.copy()
        rpy = self.drone.rpy.copy()
        vel = self.drone.xyz_dot.copy()
        rpy_dot = self.drone.rpy_dot.copy()

        for _ in range(self.frame_skip):
            rotation = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)
            thrust_world_frame = np.dot(rotation, thrust)
            force_world_frame = thrust_world_frame - np.array(
                [0, 0, self.g]) * self.drone.m

            # Note: based on X-configuration of Drone
            x_torque = (-forces[0] - forces[1] + forces[2] + forces[3]) * (
                    self.drone.L / np.sqrt(2))
            y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (
                    self.drone.L / np.sqrt(2))

            torques = np.array([x_torque, y_torque, z_torque])
            # Include Coriolis forces:
            torques = torques - np.cross(rpy_dot, np.dot(self.drone.J, rpy_dot))
            rpy_dot_dot = np.dot(self.drone.J_INV, torques)
            acc_linear = force_world_frame / self.drone.m

            vel += self.time_step * acc_linear
            rpy_dot += self.time_step * rpy_dot_dot
            pos += self.time_step * vel
            rpy += self.time_step * rpy_dot
            quat = np.array(self.bc.getQuaternionFromEuler(rpy))

        # Update drone internals
        # Note: new state information are set to PyBullet
        self.drone.xyz = pos
        self.drone.quat = quat
        self.drone.rpy = rpy
        self.drone.xyz_dot = vel
        self.drone.rpy_dot = rpy_dot
        # Set PyBullet's state
        self.bc.resetBasePositionAndOrientation(
            self.drone.body_unique_id,
            pos,
            quat
        )
        # Note: the base's velocity only stored and not used
        self.bc.resetBaseVelocity(
            self.drone.body_unique_id,
            vel,
            rpy_dot
        )
