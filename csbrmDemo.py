import time
import math
import random
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle


class Control_ACC:
    U = None
    mass = 0.5  # mass of the quadrotor
    grav = 9.81
    #### CS-BRM Data ####
    plan_text = '37.mat'
    print(plan_text)
    plan = loadmat('./'+plan_text)
    #Rnd_sample = loadmat('./random.mat')

    # DI_Discrete
    dt_plan = plan['param'][0][0][0][0][0]
    #scale = 10  # 100 Hz
    #scale = 12  # 120 Hz
    scale = 1
    dt = dt_plan/scale  # 100 Hz
    nx, ny, nu, nw = 6, 6, 3, 6
    Ak = \
    [1, 0, 0, dt, 0, 0,
     0, 1, 0, 0, dt, 0,
     0, 0, 1, 0, 0, dt,
     0, 0, 0, 1, 0, 0,
     0, 0, 0, 0, 1, 0,
     0, 0, 0, 0, 0, 1]
    Ak = np.array(Ak).reshape(6, 6)
    Bk = np.array([[dt**2/2, 0, 0], [0, dt**2/2, 0], [0, 0, dt**2/2], [dt, 0, 0], [0, dt, 0], [0, 0, dt]])
    Gk = 1 * np.sqrt(dt) * np.diag([0.02, 0.02, 0.02, 0.06, 0.06, 0.06])
    Ck = np.diag([1, 1, 1, 1, 1, 1])
    Dk = 0.01 * np.diag([1, 1, 1, 1, 1, 1])

    Covs = plan['Covs']
    Path = plan['Path'][0]
    Xbar = plan['Xall']
    EdgeControlV = plan['Vall']
    EdgeControlK = plan['Kall']
    N_idx = plan['N_idx']
    N = int(len(EdgeControlK) / 3)

    #Rnd_xhatPrior0 = Rnd_sample['Rnd_xhatPrior0']
    #Rnd_xtildePrior0 = Rnd_sample['Rnd_xtildePrior0']

    PhatPrior0 = Covs[Path[0]-1, 0]-Covs[Path[0]-1, 1]
    PtildePrior0 = Covs[Path[0]-1, 1]

    xbar0 = Xbar[:,0]
    xhatPrior0_MC = xbar0  # np.random.multivariate_normal(xbar0, PhatPrior0)
    xtildePrior0 = 0 * np.random.multivariate_normal(np.zeros((1, 6))[0], PtildePrior0)
    # xhatPrior0_MC = Rnd_xhatPrior0[:,k]
    # xtildePrior0 = Rnd_xtildePrior0[:,k]
    xbar0 = xbar0.reshape(xbar0.shape[0], 1)
    xhatPrior0_MC = xhatPrior0_MC.reshape(xhatPrior0_MC.shape[0], 1)
    xtildePrior0 = xtildePrior0.reshape(xtildePrior0.shape[0], 1)
    x0_MC = xhatPrior0_MC + xtildePrior0
    #### CS-BRM Data ####

    def MCplan(self, state_c, time_step):

        V = self.EdgeControlV
        K = self.EdgeControlK
        Xbar = self.Xbar
        N_idx = self.N_idx
        nu = self.nu

        k = math.floor(time_step/self.scale)  ###### 100 Hz

        Vc = V[k * nu: (k + 1) * nu]
        Kc = K[k * nu: (k + 1) * nu, :]
        if (k+1) in N_idx or self.U is None:
            [U, PPt, xhat_MC, z_MC] = self.computeControl_init(Xbar[:, k].reshape(len(Xbar[:, k]), 1), self.xhatPrior0_MC, self.PtildePrior0, Vc, Kc, state_c)
        else:
            [U, PPt, xhat_MC, z_MC, xhatPrior0_MC, PtildePrior0] = self.computeControl(Vc, Kc, state_c, self.U, self.PPt, self.xhat_MC, self.z_MC)
            self.xhatPrior0_MC = xhatPrior0_MC
            self.PtildePrior0 = PtildePrior0
        self.U = U
        self.PPt = PPt
        self.xhat_MC = xhat_MC
        self.z_MC = z_MC

        return U

    def computeControl_init(self, xbar0, xhatPrior0_MC, PtildePrior0, Vc, Kc, state_c):
        ny = self.ny
        Ck = self.Ck
        Dk = self.Dk

        zPrior0 = xhatPrior0_MC - xbar0
        PPtm = PtildePrior0
        # Kalman gain
        LL = PPtm.dot(Ck.T).dot(np.linalg.inv(Ck.dot(PPtm).dot(Ck.T) + Dk.dot(Dk.T)))
        # Covariance after measurement
        PPt = PPtm - LL.dot((Ck.dot(PPtm).dot(Ck.T) + Dk.dot(Dk.T))).dot(LL.T)
        # Noise sample
        v = np.random.normal(0, 1, ny)
        v = v.reshape(v.shape[0], 1)

        # Measurement
        y_MC = state_c  # [x,y,z,vx,vy,vz] measurement for Vicon
        y_MC = y_MC.reshape(y_MC.shape[0], 1) + Dk.dot(v)
        # Prior estimate
        xhatPrior_MC = xhatPrior0_MC
        # Innovation process
        ytilde_MC = y_MC - Ck.dot(xhatPrior_MC)
        # Filtered process
        xhat_MC = xhatPrior0_MC + LL.dot(ytilde_MC)
        # Feedback process
        z_MC = zPrior0 + LL.dot(ytilde_MC)

        # Control U desired acceleration
        U = Vc+ 1 * np.dot(Kc, z_MC)

        return U, PPt, xhat_MC, z_MC

    def computeControl(self, Vc, Kc, state_c, U, PPt, xhat_MC, z_MC):
        ny = self.ny
        Ak = self.Ak
        Bk = self.Bk
        Gk = self.Gk
        Ck = self.Ck
        Dk = self.Dk

        PPtm = Ak.dot(PPt).dot(Ak.T) + Gk.dot(Gk.T)
        # Kalman gain
        LL = PPtm.dot(Ck.T).dot(np.linalg.inv(Ck.dot(PPtm).dot(Ck.T) + Dk.dot(Dk.T)))
        # Covariance after measurement
        PPt = PPtm - LL.dot((Ck.dot(PPtm).dot(Ck.T) + Dk.dot(Dk.T))).dot(LL.T)
        # Noise sample
        v = np.random.normal(0, 1, ny)
        v = v.reshape(v.shape[0], 1)

        # Measurement
        y_MC = state_c  # [x,y,z,vx,vy,vz] measurement for Vicon
        y_MC = y_MC.reshape(y_MC.shape[0], 1) + Dk.dot(v)
        # Prior estimate
        xhatPrior_MC = Ak.dot(xhat_MC) + Bk.dot(U)
        # Innovation process
        ytilde_MC = y_MC - Ck.dot(xhatPrior_MC)
        # Filtered process
        xhat_MC = Ak.dot(xhat_MC) + Bk.dot(U) + LL.dot(ytilde_MC)
        # Feedback process
        z_MC = Ak.dot(z_MC) + LL.dot(ytilde_MC)

        # Control U desired acceleration
        U = Vc + 1 * np.dot(Kc, z_MC)

        return U, PPt, xhat_MC, z_MC, xhatPrior_MC, PPtm
def getSimTraj(show=False):
    '''
    run_control = Control_ACC()
    Ak = run_control.Ak
    Bk = run_control.Bk
    x_MC = [run_control.x0_MC]
    N = run_control.N
    for k in range(0, N):
        state_c = x_MC[k]
        U = run_control.MCplan(state_c, k)
        x_MC = x_MC + [np.dot(Ak, x_MC[k]) + np.dot(Bk, U)]  # + np.dot(Gk, w)]
    '''

    run_control = Control_ACC()
    Ak = run_control.Ak
    Bk = run_control.Bk
    x_MC = [run_control.x0_MC]
    scale = run_control.scale  # 100 Hz
    N = scale * run_control.N   # 100 Hz
    dt = run_control.dt
    current_time = 0
    previous_time_discrete = 0
    for k in range(0, N):
        state_c = x_MC[k]  # current state from Vicon
        U = run_control.MCplan(state_c, k)
        x_MC = x_MC + [np.dot(Ak, x_MC[k]) + np.dot(Bk, U)]  # + np.dot(Gk, w)]

        '''
        if current_time == 0:
            state_c = x_MC[k]  # current state from Vicon
            U = run_control.MCplan(state_c, k)
            x_MC = x_MC + [np.dot(Ak, x_MC[k]) + np.dot(Bk, U)]  # + np.dot(Gk, w)]
            current_time += dt

        elif current_time - previous_time_discrete > 0.09999:
            state_c = x_MC[k]  # current state from Vicon
            U = run_control.MCplan(state_c, k)
            x_MC = x_MC + [np.dot(Ak, x_MC[k]) + np.dot(Bk, U)]  # + np.dot(Gk, w)]
            previous_time_discrete += 0.1
            current_time += 0.1
        '''

    t1 = dt * np.arange(len(x_MC))
    X_MC = np.array(x_MC)
    if (show):
        print(np.diff(X_MC[:,0])/0.1)
        # plt.figure()
        plt.plot(t1, X_MC[:, 0], color='r', linestyle='--')
        plt.plot(t1, X_MC[:, 1], color='g', linestyle='--')
        plt.plot(t1, X_MC[:, 2], color='b', linestyle='--')
        plt.legend(["x", "y", "z"])
        plt.xlabel('Time [s]')
        plt.ylabel('Position [m]')
        plt.show()
    return (X_MC[:,0].flatten(), X_MC[:,1].flatten(), X_MC[:,2].flatten())


if __name__=='__main__':
    traj = getSimTraj(True)
    with open('csbrm_traj.p','wb') as f:
        pickle.dump(traj,f)
