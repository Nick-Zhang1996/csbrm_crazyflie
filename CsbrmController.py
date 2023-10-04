from CSBRMlanding import Control_ACC
import numpy as np
from PidController import PidController
from math import degrees,radians
from common import *

class CsbrmController:
    def __init__(self):
        dt = 1/120.0
        self.Tf = 8.5
        self.csbrm = Control_ACC()
        self.offset = 0
        # NOTE offset height for safety
        # init_pos_planner_frame = 1,1,2.5
        # NED frame
        self.init_pos = (1,-1,-(2.5-self.offset))
        # physical properties
        self.g = g = 9.81
        self.m = 40e-3
        self.max_thrust = 62e-3 * g
        self.yaw_pid = PidController(2,0,0,dt,0,20)
        self.log = [0,0,0]

        self.last_timestep = -1
        self.old_timestep = -1
    def getInitialPosition(self):
        return self.init_pos

    # state = (x,y,z,vx,vy,vz,rx,ry,rz) in VICON space, with z inverted
    # VICON -> conventional
    # x-> x
    # y-> -y
    # z-> -z
    # if control is finished, return None
    #(target_roll_deg, target_pitch_deg, target_yawrate_deg_s, target_thrust_raw) = ret
    # reference: cfcontroller:control()
    def control(self,t, drone_state):
        if (t > self.Tf):
            return None
        (x,y,z,vx,vy,vz,rx,ry,rz) = drone_state

        # add height offset  for safety
        # lie to planner that quadcopter is higher
        z = z-self.offset

        # produce state in planner ref frame
        state_planner = (x, -y, -z, vx, -vy, -vz)
        # desired acceleration
        #time_step = int(t / 0.1)
        time_step = int(t * 50)
        if (time_step > self.old_timestep):
            self.acc_des_planner = self.csbrm.MCplan(np.array(state_planner), time_step)
            self.old_timestep = time_step
        # append controller output
        self.log = list(self.acc_des_planner.flatten())
        (ax_planner, ay_planner, az_planner) = self.acc_des_planner.flatten()
        acc_des = np.array((ax_planner, -ay_planner, -az_planner))
        gravity = np.array((0,0,self.m*self.g))
        # WARNING NOTE unbounded
        Fdes = -gravity + self.m * acc_des
        # max: 15m/s2
        '''
        if (np.linalg.norm(Fdes) > self.max_thrust*200):
            print_info('requested acc', np.linalg.norm(acc_des))
            print_info('requested force', np.linalg.norm(Fdes))
            print_warning('exceeding maximium allowable acceleration')
            return None
        '''

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

