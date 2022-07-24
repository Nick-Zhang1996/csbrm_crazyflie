# simple controller
import numpy as np
from PidController import PidController
from math import radians,degrees
from common import *
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from time import time

class SampleController:
    def __init__(self,dt=1/120.0,time_scale=1.0):
        self.kp = 1.0
        self.kv = 1.0
        self.dt = dt
        self.time_scale = time_scale

        # total experiment time
        self.Tf = 20
        # period of periodic trajectory
        self.soft_start_T = 10
        self.soft_start_duration = 5
        self.T = 10

        self.yaw_pid = PidController(2,0,0,dt,0,20)

        # physical properties
        self.g = g = 9.81
        self.m = 40e-3
        self.max_thrust = 62e-3 * g
        self.getTrajectory = self._getLoiterTrajectory
        self.getTrajectory = self._getFlowerTrajectory

    # flower shape
    def _getFlowerTrajectory(self, t, der=0):
        # initially quad is stationary
        # if a small T is used, vehicle won't be able to keep up
        # so we use a soft start
        if (t < self.soft_start_duration):
            T = self.soft_start_T + t/self.soft_start_duration * (self.T - self.soft_start_T)
        else:
            T = self.T
        R = 1.0
        A = 0.2
        e = 1e-3
        x = lambda t: np.cos(t/T * 2*np.pi) * (R + A*np.cos(t/T*4* 2*np.pi))
        y = lambda t: np.sin(t/T * 2*np.pi) * (R + A*np.cos(t/T*4* 2*np.pi))
        z = lambda t: -0.5

        if (der == 0):
            return np.array((x(t),y(t),z(t)))

        deri = lambda fun,t: (fun(t+e) - fun(t-e))/(2*e)
        if (der == 1):
            return np.array((deri(x,t),deri(y,t),deri(z,t)))

        dderi = lambda fun,t: (fun(t+e) -2*fun(t) + fun(t-e))/(e*e)
        if (der == 2):
            return np.array((dderi(x,t),dderi(y,t),dderi(z,t)))

    # zero / stay in place
    def _getLoiterTrajectory(self, t, der=0):
        if (t>self.Tf or t < 0):
            return None
        if (der == 0):
            return np.array((0,0,-0.3))

        if (der == 1):
            return np.array((0,0,0))

        if (der == 2):
            return np.array((0,0,0))

    # get trajectory or it's derivative
    # 3d veector function parameterized by t
    # simple traj
    def _getSimpleTrajectory(self, t, der=0):
        # x = y = -2 (t/T)3 + 3(t/T)2
        # z = -0.3
        # dxdt = dydt = -6**2 / (T**3) + 6 * t / (T**2)
        # xy(0) = 0
        # xy(T) = 1.0

        if (t>self.T or t < 0):
            return None

        T = self.T
        z = -0.3
        dzdt = 0.0
        ddzdt = 0.0
        if (der == 0):
            x = y = -2 * (t/T)**3 + 3*(t/T)**2
            return np.array((x,y,z))

        if (der == 1):
            dxdt = dydt = -6**2 / (T**3) + 6 * t / (T**2)
            return np.array((dxdt,dydt,dzdt))

        if (der == 2):
            ddxdt = ddydt = -12 * t / (T**3) + 6/(T**2)
            return np.array((ddxdt,ddydt,ddzdt))

    # t: time, elapsed since trajectory start
    # drone_state: (x,y,z,vx,vy,vz, rx,ry,rz) of drone
    # ddrdt: second derivative of trajectory
    def control(self, t, drone_state, r_des=None, drdt_des = None, ddrdt_des=None):
        if (t > self.Tf):
            return None
        (x,y,z,vx,vy,vz,rx,ry,rz) = drone_state
        r = np.array((x,y,z))
        drdt = np.array((vx,vy,vz))
        gravity = np.array((0,0,self.m*self.g))
        if (r_des is None):
            r_des = np.array(self.getTrajectory(t, der=0))
            drdt_des = np.array(self.getTrajectory(t, der=1))
            ddrdt_des = np.array(self.getTrajectory(t, der=2))
        ep = r - r_des
        ev = drdt - drdt_des
        Fdes = (-self.kp*ep - self.kv*ev)*self.m*self.g - gravity + self.m*ddrdt_des

        # Fdes = R @ [0,0,-T]
        T_des = np.linalg.norm(Fdes)

        #Thrust_vec = np.array([0,0,-1])
        #axis = ( Fdes/np.linalg.norm(Fdes) + Thrust_vec / np.linalg.norm(Thrust_vec))
        #r_vec = axis / np.linalg.norm(axis) * np.pi
        #r = R.from_rotvec(r_vec)
        #yaw_des, pitch_des, roll_des = r.as_euler('ZYX')
        roll_des = np.arcsin(Fdes[1]/T_des)
        pitch_des = np.arctan2(-Fdes[0], -Fdes[2])
        yaw_des = 0.0

        yaw_diff = yaw_des - rz
        yaw_diff = (yaw_diff + np.pi)%(2*np.pi) - np.pi
        target_yawrate_deg_s = self.yaw_pid.control(degrees(rz + yaw_diff), degrees(rz))
        target_roll_deg = degrees(roll_des)
        target_pitch_deg = degrees(pitch_des)
        target_thrust_raw = int(T_des / self.max_thrust * 65535)
        self.debug = [target_roll_deg, target_pitch_deg, degrees(yaw_des)]
        return (target_roll_deg, target_pitch_deg, target_yawrate_deg_s, target_thrust_raw)

if __name__=="__main__":
    # examine trajectory
    main = SampleController(0.01)
    g = 9.81

    rpy_data = []
    thrust_data = []
    pos_data = []
    vel_data = []
    acc_data = []

    for t in np.linspace(0,main.Tf):
        r = main.getTrajectory(t,der=0)
        pos_data.append(r)
        dr = main.getTrajectory(t,der=1)
        vel_data.append(dr)
        ddr = main.getTrajectory(t,der=2)
        acc_data.append(ddr)

        drone_state = np.hstack([r,dr,np.zeros(3)])
        ret = main.control(0, drone_state , r,dr,ddr)
        (target_roll_deg, target_pitch_deg, target_yawrate_deg_s, target_thrust_raw) = ret
        rpy_data.append(main.debug)
        thrust_data.append((float(target_thrust_raw)/65536)*main.max_thrust/main.m/main.g)

    rpy_data = np.array(rpy_data)
    thrust_data = np.array(thrust_data)
    pos_data = np.array(pos_data)
    vel_data = np.array(vel_data)
    acc_data = np.array(acc_data)

    print("pos")
    plt.plot(pos_data)
    plt.show()

    print("vel")
    plt.plot(vel_data)
    plt.show()

    print("acc")
    plt.plot(acc_data)
    plt.show()

    print("rpy")
    plt.plot(rpy_data)
    plt.show()

    print("thrust")
    plt.plot(thrust_data)
    plt.show()





