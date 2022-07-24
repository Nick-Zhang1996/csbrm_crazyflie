from common import ndarray, angular_difference
import numpy as np

class PidController:
    def __init__(self, p, i, d, dt, integral_limit, lpf_cutoff_freq):
        self.__p = p
        self.__i = i
        self.__d = d
        self.__integral = 0
        self.__last_error = 0
        self.__last_x0 = 0
        self.__dt = dt
        self.__integral_limit = integral_limit
        self.__alpha = (2 * np.pi * self.__dt * lpf_cutoff_freq) / (2 * np.pi * self.__dt * lpf_cutoff_freq + 1)

    # dxdt: allow user to provide derivative externally, i.e. from a gyro
    # x0: setpoint
    # x: measurement
    def control(self, x0, x, dxdt=None, debug=False):
        error = x0 - x
        # Low pass filter: larger lpf freq => error = error; small lpf freq => error = last error.
        error = (1 - self.__alpha) * self.__last_error + self.__alpha * error
        # P:
        p_term = self.__p * error
        # I:
        self.__integral += error * self.__dt
        self.__integral = np.clip(self.__integral, -self.__integral_limit, self.__integral_limit)
        i_term = self.__i * self.__integral
        # D:
        if dxdt is None:
            derivative = (error - self.__last_error) / self.__dt
        else:
            derivative = (x0 - self.__last_x0) / self.__dt - dxdt
        self.__last_x0 = x0
        d_term = self.__d * derivative
        # Update.
        self.__last_error = error
        if debug:
            return p_term + i_term + d_term,{'p_term':p_term,'i_term':i_term,'d_term':d_term,'err':error}
        else:
            return p_term + i_term + d_term

class CrazyfliePidController:
    def __init__(self, dt):
        self.dt = dt
        # Our crazyflie weighs 33grams
        self.mass = 33 / 1000
        self.g = 9.81

        p = np.rad2deg(800)
        i = 0
        d = np.rad2deg(120)
        self.pitch_controller = PidController(p, i, d, self.dt, np.deg2rad(33.3), 30)
        self.roll_controller = PidController(p, i, d, self.dt, np.deg2rad(33.3), 30)

        d = np.rad2deg(2)
        self.pitch_rate_controller = PidController(0, 0, d, self.dt, np.deg2rad(33.3), 30)
        self.roll_rate_controller = PidController(0, 0, d, self.dt, np.deg2rad(33.3), 30)

        p = np.rad2deg(6)
        i = np.rad2deg(0)
        d = np.rad2deg(0.35)
        self.yaw_controller = PidController(p, i, d, self.dt, 33.3, 30)

        p = 120.0
        i = 0
        d = 0
        self.yaw_rate_controller = PidController(p, i, d, self.dt, 33.3, 30)

        p = 17000
        i = 500
        d = 1700
        self.altitude_controller = PidController(p, i, d, self.dt, 10000, 30)

        p = 0.25
        i = 0
        d = 0.01
        self.x_controller = PidController(p, i, d, self.dt, 0, 10)
        self.y_controller = PidController(p, i, d, self.dt, 0, 10)

    # Input is in kg.
    def __mass_to_control(self, m):
        return m / (15 / 1000) * 65536

    def attitude_control(self, rpy, omega_body, target):
        # target: target roll, pitch, yaw (in radians) + base thrusts (0 - 65535).
        # rpy: in radians.
        # omega_body: in radians/seconds.
        roll, pitch, yaw = rpy
        omega_x_rate, omega_y_rate, omega_z_rate = omega_body
        target_roll, target_pitch, target_yaw, base_thrust = target

        # Pitch.
        pitch_moment = self.pitch_controller.control(target_pitch, pitch, omega_y_rate)
        pitch_moment += self.pitch_rate_controller.control(0, omega_y_rate)
        pitch_moment = np.clip(pitch_moment, -20000, 20000)

        # Roll.
        roll_moment = self.roll_controller.control(target_roll, roll, omega_x_rate)
        roll_moment += self.roll_rate_controller.control(0, omega_x_rate)
        roll_moment = np.clip(roll_moment, -20000, 20000)

        # Yaw.
        yaw_rate_rads = omega_z_rate
        diff = angular_difference(target_yaw, yaw)
        target_yaw_rate_rads = self.yaw_controller.control(yaw + diff, yaw, omega_z_rate, False)
        yaw_moment = self.yaw_rate_controller.control(target_yaw_rate_rads, omega_z_rate)
        yaw_moment = np.clip(yaw_moment, -20000, 20000)

        # Follow crazyflie convention.
        pitch_moment /= 2.0
        roll_moment /= 2.0
        yaw_moment /= 2.0

        # Contribution.
        m1 = int(np.clip(yaw_moment - roll_moment + pitch_moment + base_thrust, 0, 0xFFFF))
        m2 = int(np.clip(-yaw_moment - roll_moment - pitch_moment + base_thrust, 0, 0xFFFF))
        m3 = int(np.clip(yaw_moment + roll_moment - pitch_moment + base_thrust, 0, 0xFFFF))
        m4 = int(np.clip(-yaw_moment + roll_moment + pitch_moment + base_thrust, 0, 0xFFFF))

        # Safety.
        if base_thrust < 1000:
            thrusts = [0, 0, 0, 0]
        else:
            thrusts = [m1, m2, m3, m4]
        return thrusts

    def position_control(self, xyz, xyz_dot, rpy, omega_body, target):
        # xyz: in meters.
        # xyz_dot: in meters/seconds.
        # rpy: in radians.
        # omega_body: in radians/seconds.
        # target: x (meter), y (meter), yaw (radinas), altitude (meter).
        target_x, target_y, target_yaw, target_altitude = target
        # Compute target roll and pitch.
        yaw_rad = rpy[2]
        x, y, _ = xyz
        x_dot, y_dot, _ = xyz_dot
        offset = ndarray([target_x - x, target_y - y])
        c_yaw, s_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
        R = ndarray([[c_yaw, s_yaw], [-s_yaw, c_yaw]])
        offset = R @ offset
        x_dot_body, y_dot_body = R @ ndarray([x_dot, y_dot])
        target_x_body, target_y_body = offset

        target_roll = self.y_controller.control(0, -target_y_body, x_dot_body, False)
        target_pitch = -self.x_controller.control(0, -target_x_body, y_dot_body, False)
        target_roll = np.clip(target_roll, np.deg2rad(-5), np.deg2rad(5))
        target_pitch = np.clip(target_pitch, np.deg2rad(-5), np.deg2rad(5))

        altitude = -xyz[2]
        altitude_rate = -xyz_dot[2]

        if target_altitude < -0.5:
            base_thrust = 0
        else:
            base_thrust = self.__mass_to_control(self.mass / 4) + \
                self.altitude_controller.control(target_altitude, altitude, altitude_rate, False)

        return self.attitude_control(rpy, omega_body, [target_roll, target_pitch, target_yaw, base_thrust])

if __name__ == '__main__':
    pass
