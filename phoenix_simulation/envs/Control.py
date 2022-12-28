import abc
import numpy as np
import pybullet as pb
from pybullet_utils import bullet_client

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi

# ==============================================================================
#   Settings from the CrazyFlie Firmware, see:
#   https://github.com/bitcraze/crazyflie-firmware/blob/master/src/modules/src/attitude_pid_controller.c
# ==============================================================================

PID_ROLL_RATE_KP = 250.0  # default: 250.0
PID_ROLL_RATE_KI = 500.0  # default: 500.0
PID_ROLL_RATE_KD = 2.5  # default: 2.50
PID_ROLL_RATE_INTEGRATION_LIMIT = 33.3

PID_PITCH_RATE_KP = 250.0  # default: 250.0
PID_PITCH_RATE_KI = 500.0  # default: 500.0
PID_PITCH_RATE_KD = 2.5  # default: 2.50
PID_PITCH_RATE_INTEGRATION_LIMIT = 33.3

PID_YAW_RATE_KP = 120.0
PID_YAW_RATE_KI = 16.7
PID_YAW_RATE_KD = 0.0
PID_YAW_RATE_INTEGRATION_LIMIT = 166.7


class Control(object):
    r"""Parent class for control objects."""

    def __init__(
            self,
            drone,
            bc: bullet_client.BulletClient,
            time_step: float,  # 1 / sim_frequency
            frame_skip: int,
    ):
        self.drone = drone
        self.bc = bc
        self.control_counter = 0
        self.time_step = time_step
        self.frame_skip = frame_skip

    def __call__(self, *args, **kwargs):
        return self.act(*args, **kwargs)

    @abc.abstractmethod
    def act(self, action, **kwargs):
        r"""Action to PWM signal."""
        raise NotImplementedError

    def reset(self):
        r"""Reset the control classes.

        A general use counter is set to zero.
        """
        self.control_counter = 0


class PWM(Control):
    r"""Class for direct PWM motor control."""

    def act(self, action, **kwargs):
        r"""Action to PWM signal."""
        clipped_action = np.clip(action, -1, 1)
        PWMs = 2**15 + clipped_action * 2**15  # PWM in [0, 65535]
        return PWMs


# class PID(Control):
#     def __init__(
#             self,
#             drone,
#             bc: bullet_client.BulletClient,
#             time_step: float,
#             kp: float,
#             ki: float,
#             kd: float,
#     ):
#         super(PID, self).__init__(
#             drone=drone,
#             bc=bc,
#             time_step=time_step
#         )


class AttitudeRate(Control):
    def __init__(
            self,
            drone,
            bc: bullet_client.BulletClient,
            time_step: float,  # 1 / sim_frequency
            frame_skip: int,
    ):
        super(AttitudeRate, self).__init__(
            drone=drone,
            bc=bc,
            time_step=time_step,
            frame_skip=frame_skip
        )
        # self.state = None
        self.integral = np.zeros(3)
        self.last_error = np.zeros(3)

        # Attitude Rate parameters:
        self.kp_att_rate = np.array(
            [PID_ROLL_RATE_KP, PID_PITCH_RATE_KP, PID_YAW_RATE_KP])
        self.ki_att_rate = np.array(
            [PID_ROLL_RATE_KI, PID_PITCH_RATE_KI, PID_YAW_RATE_KI])
        self.kd_att_rate = np.array(
            [PID_ROLL_RATE_KD, PID_PITCH_RATE_KD, PID_YAW_RATE_KD])

        self.rpy_rate_integral_limits = np.array(
            [PID_ROLL_RATE_INTEGRATION_LIMIT,
             PID_PITCH_RATE_INTEGRATION_LIMIT,
             PID_YAW_RATE_INTEGRATION_LIMIT])

        self.reset()

    def act(self, action, **kwargs):
        """Action to PWM signal."""
        clipped_action = np.clip(action, -1, 1)
        # Action = [thrust, roll_dot, pitch_dot, yaw_dot]
        thrust = 2**15 + clipped_action[0] * 2**15
        rpy_dot_target = clipped_action[1:4] * 2 * np.pi
        rpy_factors = self.compute_control(rpy_dot_target)

        assert rpy_factors.shape == (3, )
        control_roll, control_pitch, control_yaw = rpy_factors

        def limitThrust(pwm_value):
            """Limit PWM signal."""
            return np.clip(pwm_value, 0, 2**16)

        """Convert PID output to motor power signal (uint16).
          #ifdef QUAD_FORMATION_X
            int16_t r = control->roll / 2.0f;
            int16_t p = control->pitch / 2.0f;
            motorPower.m1 = limitThrust(control->thrust - r + p + control->yaw);
            motorPower.m2 = limitThrust(control->thrust - r - p - control->yaw);
            motorPower.m3 =  limitThrust(control->thrust + r - p + control->yaw);
            motorPower.m4 =  limitThrust(control->thrust + r + p - control->yaw);
        """
        r = control_roll / 2.0
        p = control_pitch / 2.0
        PWMs = np.empty(4)
        PWMs[0] = limitThrust(thrust - r - p - control_yaw)
        PWMs[1] = limitThrust(thrust - r + p + control_yaw)
        PWMs[2] = limitThrust(thrust + r + p - control_yaw)
        PWMs[3] = limitThrust(thrust + r - p + control_yaw)
        return PWMs

    def compute_control(
            self,
            rpy_dot_target
    ):
        """Computes the PID control action (as PWMs) for a single drone."""
        dt = self.time_step * self.frame_skip
        # Transform values to local frame of drone
        R = np.array(pb.getMatrixFromQuaternion(self.drone.quat)).reshape((3, 3))
        rpy_dot_local_frame = R @ self.drone.rpy_dot

        # Note: PyBullet calculates in rad whereas the firmware takes degrees
        error = (rpy_dot_target - rpy_dot_local_frame) * RAD2DEG
        # print(f'R: {error[0]:0.2f}\t P: {error[1]:0.2f}\t Y: {error[2]:0.2f}')
        # print(f'IR:{integ[0]:0.2f}\tIP: {integ[1]:0.2f}\tIY: {integ[2]:0.2f}')
        derivative = (error - self.last_error) / dt
        self.last_error = error
        self.integral += error * dt
        # limit integral values
        self.integral = np.clip(self.integral, -self.rpy_rate_integral_limits,
                                self.rpy_rate_integral_limits)
        # print('self.integral:', self.integral)
        PWMs = self.kp_att_rate * error + self.ki_att_rate * self.integral \
                         + self.kd_att_rate * derivative
        return PWMs

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        # Initialized PID control variables
        self.integral = np.zeros(3)
        self.last_error = np.zeros(3)
