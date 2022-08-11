# 10Hz benchmark controller
import numpy as np
from PidController import PidController
from math import radians,degrees
from common import *
from scipy.spatial.transform import Rotation as R
#from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from time import time
import pickle

class BenchmarkController:
    def __init__(self,dt=1/120.0,time_scale=1.0):
        self.kp = 1.0
        self.kv = 1.0
        self.dt = dt
        self.time_scale = time_scale

        # total experiment time
        self.Tf = 8.5

        self.yaw_pid = PidController(2,0,0,dt,0,20)

        # physical properties
        self.g = g = 9.81
        self.m = 40e-3
        self.max_thrust = 62e-3 * g

        # load reference trajectory
        with open('csbrm_traj.p','rb') as f:
            self.traj = pickle.load(f)
        # make traj = [x(t), y(t), z(t)]
        traj_len = (len(self.traj[0])-1)*0.1
        tt = np.linspace(0,traj_len,len(self.traj[0]))
        #self.traj_x_fun = interp1d(tt,self.traj[0], kind='cubic', bounds_error=False, fill_value='extrapolate')
        #self.traj_y_fun = interp1d(tt,self.traj[1], kind='cubic', bounds_error=False, fill_value='extrapolate')
        #self.traj_z_fun = interp1d(tt,self.traj[2], kind='cubic', bounds_error=False, fill_value='extrapolate')
        self.traj_x_fun = UnivariateSpline(tt,self.traj[0], k=4,s=0)
        self.traj_y_fun = UnivariateSpline(tt,self.traj[1], k=4,s=0)
        self.traj_z_fun = UnivariateSpline(tt,self.traj[2], k=4,s=0)
        # verify trajectory
        tt = np.linspace(0,traj_len,300)
        xx = self.traj_x_fun(tt)
        yy = self.traj_y_fun(tt)
        zz = self.traj_z_fun(tt)
        plt.plot(tt, xx, color='r', linestyle='--')
        plt.plot(tt, yy, color='g', linestyle='--')
        plt.plot(tt, zz, color='b', linestyle='--')

        tt = np.linspace(0,traj_len,len(self.traj[0]))
        plt.plot(tt, self.traj[0], color='r')
        plt.plot(tt, self.traj[1], color='g')
        plt.plot(tt, self.traj[2], color='b')
        plt.show()

    def getInitialPosition(self):
        return self.getTrajectory(0)

    # flower shape
    def getTrajectory(self, t, der=0):
        # initially quad is stationary
        e = 1e-3
        x = self.traj_x_fun
        y = self.traj_y_fun
        z = self.traj_z_fun

        if (der == 0):
            return np.array((x(t),y(t),z(t)))

        '''
        deri = lambda fun,t: (fun(t+e) - fun(t-e))/(2*e)
        if (der == 1):
            return np.array((deri(x,t),deri(y,t),deri(z,t)))

        dderi = lambda fun,t: (fun(t+e) -2*fun(t) + fun(t-e))/(e*e)
        if (der == 2):
            return np.array((dderi(x,t),dderi(y,t),dderi(z,t)))
        '''

        if (der == 1 or der == 2):
            deri_x = x.derivative(n=der)
            deri_y = y.derivative(n=der)
            deri_z = z.derivative(n=der)
            return np.array((deri_x(t), deri_y(t), deri_z(t)))


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
    main = BenchmarkController(0.01)
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





