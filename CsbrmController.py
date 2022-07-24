from quadsim import Control_ACC
import numpy as np

class CsbrmController:
    def __init__(self):
        self.csbrm = Control_ACC()
        self.init_pos = (1,1,2.5)
        # physical properties
        self.g = g = 9.81
        self.m = 40e-3
        self.max_thrust = 62e-3 * g
    def getInitialPosition(self):
        return self.init_pos

    # state = (x,y,z,vx,vy,vz,rx,ry,rz) in VICON space, with z inverted
    # VICON -> conventional
    # x-> -y
    # y->  x
    # z-> -z
    # if control is finished, return None
    #(target_roll_deg, target_pitch_deg, target_yawrate_deg_s, target_thrust_raw) = ret
    # reference: cfcontroller:control()
    def control(self,t, drone_state):
        (x,y,z,vx,vy,vz,rx,ry,rz) = drone_state

        # produce state in planner ref frame
        state_planner = (y, -x, -z, vy, -vx, -vz)
        # desired acceleration
        time_step = int(t / 0.1)
        acc_des_planner = self.csbrm.MCplan(state, time_step)
        (ax_planner, ay_planner, az_planner)
        acc_des = np.array((-y, x, -z))
        gravity = np.array((0,0,self.m*self.g))
        Fdes = -gravity + self.m * acc_des


        # thrust
        T_des = np.linalg.norm(Fdes)
        roll_des = np.arcsin(Fdes[1]/T_des)
        pitch_des = np.arctan2(-Fdes[0], -Fdes[2])
        yaw_des = 0.0

        yaw_diff = yaw_des - rz
        yaw_diff = (yaw_diff + np.pi)%(2*np.pi) - np.pi
        target_yawrate_deg_s = self.yaw_pid.control(degrees(rz + yaw_diff), degrees(rz))
        target_roll_deg = degrees(roll_des)
        target_pitch_deg = degrees(pitch_des)
        target_thrust_raw = int(T_des / self.max_thrust * 65535)
        return (target_roll_deg, target_pitch_deg, target_yawrate_deg_s, target_thrust_raw)

