import time
import math
import random
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle
from heapq import heappop, heappush


class Control_ACC:
    def __init__(self):
        #### CS-BRM Data ####
        plan = loadmat('graphFinal.mat')
        self.Nodes = plan['Nodes']
        self.ChildM = plan['ChildM']
        self.EdgesCost = plan['EdgesCost']
        self.Covs = plan['Covs']
        self.EdgeControlV = plan['EdgeControlV']
        self.EdgeControlK = plan['EdgeControlK']
        self.EdgeTraj = plan['EdgeTraj']
        #### CS-BRM Data ####

        # DI_Discrete
        dt_plan = plan['param'][0][0][0][0][0]
        #scale = 10  # 100 Hz
        self.scale = 1
        dt = dt_plan/self.scale  # 100 Hz
        self.dt = dt
        self.nx, self.ny, self.nu, self.nw = 6, 6, 3, 6
        Ak = \
        [1, 0, 0, dt, 0, 0,
        0, 1, 0, 0, dt, 0,
        0, 0, 1, 0, 0, dt,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1]
        self.Ak = np.array(Ak).reshape(6, 6)
        self.Bk = np.array([[dt**2/2, 0, 0], [0, dt**2/2, 0], [0, 0, dt**2/2], [dt, 0, 0], [0, dt, 0], [0, 0, dt]])
        self.Gk = 0.5 * np.sqrt(dt) * np.diag([0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
        self.Ck = np.diag([1, 1, 1, 1, 1, 1])
        self.Dk = np.diag([1, 1, 1, 1, 1, 1])

        self.U = None
        self.init = None
        self.goal = None

    def set_startgoal(self, ini_idx=1):
        self.init = ini_idx - 1
        landingPoint_idx = [240,241,242]
        idx = np.random.choice(len(landingPoint_idx), 1)
        while landingPoint_idx[idx[0]] == ini_idx:
            idx = np.random.choice(len(landingPoint_idx), 1)
        self.goal = landingPoint_idx[idx[0]] - 1

    def Astar(self):
        start, goal = self.init, self.goal
        came_from = {start: None}
        g_cost = {start: 0}
        queue = [(0, start)]  # g_cost, vertex

        vertex_count = 0
        while queue:
            current_g, current = heappop(queue)
            if current == goal:
                print('Astar vertex_count', vertex_count)
                return self.reconstruct_path(came_from, goal), g_cost[goal]
            if current in g_cost and current_g > g_cost[current]:  # current already expanded with a lower cost-to-come
                continue  # queue can have repeated vertex with different cost
            vertex_count += 1
            for i in range(int(self.Nodes[current][6])):
                new = self.ChildM[current][i]-1
                new_g = current_g + self.EdgesCost[current][i]
                if new not in g_cost or new_g < g_cost[new]:
                    g_cost[new] = new_g
                    came_from[new] = current
                    heappush(queue, (new_g, new))
        return None, None

    def reconstruct_path(self, came_from, current):
        start = self.init
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        self.path = path
        return path

    def getPlan(self):
        Kall = [[0, 0, 0, 0, 0, 0]]
        Vall = [[0]]
        Xall = [[0], [0], [0], [0], [0], [0]]
        N_idx = []
        for i in range(len(self.path)-1):
            N_idx = np.append(N_idx, np.size(Xall,1))
            indx = np.where(self.ChildM[self.path[i]] == np.array([self.path[i+1]])+1)[0][0]
            V = self.EdgeControlV[self.path[i]][indx]
            K = self.EdgeControlK[self.path[i]][indx]
            Xbar = self.EdgeTraj[path[i]][indx]
            Vall = np.append(Vall, V, axis=0)
            for j in range(int(np.size(K, 0)/3)):
                Kall = np.append(Kall, K[3*j:3*(j+1), 6*j:6*(j+1)], axis=0)
            Xall = np.append(Xall, Xbar[:, :-1], axis=1)
        Xall = np.append(Xall, Xbar[:, -1].reshape(6, 1), axis=1)
        self.N_idx = N_idx
        self.Vall = np.delete(Vall, 0, 0)
        self.Kall = np.delete(Kall, 0, 0)
        self.Xall = np.delete(Xall, 0, 1)

        PhatPrior0 = self.Covs[self.path[0], 0] - self.Covs[self.path[0], 1]
        self.PtildePrior0 = self.Covs[self.path[0] , 1]
        xbar0 = self.Xall[:, 0]
        xhatPrior0_MC = xbar0  # np.random.multivariate_normal(xbar0, PhatPrior0)
        self.xhatPrior0_MC = xhatPrior0_MC.reshape(xhatPrior0_MC.shape[0], 1)


    def MCplan(self, state_c, time_step):
        Xbar = self.Xall
        nu = self.nu
        k = math.floor(time_step/self.scale)  ###### 100 Hz

        Vc = self.Vall[k * nu: (k + 1) * nu]
        Kc = self.Kall[k * nu: (k + 1) * nu, :]
        if (k+1) in self.N_idx or self.U is None:
            [U, PPt, xhat_MC, z_MC] = self.computeControl_init(Xbar[:, k].reshape(len(Xbar[:, k]), 1), self.xhatPrior0_MC, self.PtildePrior0, Vc, Kc, state_c)
        else:
            [U, PPt, xhat_MC, z_MC, PtildePrior0] = self.computeControl(Vc, Kc, state_c, self.U, self.PPt, self.xhat_MC, self.z_MC)
            if (k+2) in self.N_idx:
                xhatPrior0_MC = self.Ak.dot(xhat_MC) + self.Bk.dot(U)
                self.xhatPrior0_MC = xhatPrior0_MC
            self.PtildePrior0 = PtildePrior0
        self.U = U
        self.PPt = PPt
        self.xhat_MC = xhat_MC
        self.z_MC = z_MC

        return U

    def computeControl_init(self, xbar0, xhatPrior0_MC, PtildePrior0, Vc, Kc, state_c):
        Ck = self.Ck
        dist = np.linalg.norm(state_c[0:2].T - [[0.75, 2], [1.25, 2], [1.25, 0.75], [2.25, 1.25], [2.75, 0.75], [3, 2]], axis=1)
        if min(dist) < 0.6:
            Dk = 0.08 * self.Dk
        else:
            Dk = 0.04 * self.Dk

        zPrior0 = xhatPrior0_MC - xbar0
        PPtm = PtildePrior0
        # Kalman gain
        LL = PPtm.dot(Ck.T).dot(np.linalg.inv(Ck.dot(PPtm).dot(Ck.T) + Dk.dot(Dk.T)))
        # Covariance after measurement
        PPt = PPtm - LL.dot((Ck.dot(PPtm).dot(Ck.T) + Dk.dot(Dk.T))).dot(LL.T)
        # Noise sample
        v = 0 * np.random.normal(0, 1, self.ny)
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
        U = Vc + 1.5 * np.dot(Kc, np.append(z_MC[0:3], [[0],[0],[0]], axis=0)) + 1 * np.dot(Kc, np.append([[0],[0],[0]], z_MC[3:6], axis=0))

        return U, PPt, xhat_MC, z_MC

    def computeControl(self, Vc, Kc, state_c, U, PPt, xhat_MC, z_MC):
        Ak = self.Ak
        Bk = self.Bk
        Gk = self.Gk
        Ck = self.Ck
        dist = np.linalg.norm(state_c[0:2].T - [[0.75, 2], [1.25, 2], [1.25, 0.75], [2.25, 1.25], [2.75, 0.75], [3, 2]], axis=1)
        if min(dist) < 0.6:
            Dk = 0.08 * self.Dk
        else:
            Dk = 0.04 * self.Dk

        PPtm = Ak.dot(PPt).dot(Ak.T) + Gk.dot(Gk.T)
        # Kalman gain
        LL = PPtm.dot(Ck.T).dot(np.linalg.inv(Ck.dot(PPtm).dot(Ck.T) + Dk.dot(Dk.T)))
        # Covariance after measurement
        PPt = PPtm - LL.dot((Ck.dot(PPtm).dot(Ck.T) + Dk.dot(Dk.T))).dot(LL.T)
        # Noise sample
        v = 0 * np.random.normal(0, 1, self.ny)
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
        U = Vc + 1.5 * np.dot(Kc, np.append(z_MC[0:3], [[0],[0],[0]], axis=0)) + 1 * np.dot(Kc, np.append([[0],[0],[0]], z_MC[3:6], axis=0))

        return U, PPt, xhat_MC, z_MC, PPtm

    def getSimTraj(self, show=False):
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

        Ak = self.Ak
        Bk = self.Bk
        xtildePrior0 = 0 * np.random.multivariate_normal([0,0,0,0,0,0], self.PtildePrior0)
        xtildePrior0 = xtildePrior0.reshape(xtildePrior0.shape[0], 1)
        self.x0_MC = 1.1 * self.xhatPrior0_MC + xtildePrior0
        x_MC = [self.x0_MC]
        scale = self.scale  # 100 Hz
        N = scale * int(len(self.Kall) / 3)  # 100 Hz
        dt = self.dt
        current_time = 0
        previous_time_discrete = 0
        for k in range(0, N):
            state_c = x_MC[k]  # current state from Vicon
            U = self.MCplan(state_c, k)
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
            # plt.figure()
            plt.plot(t1, X_MC[:, 0], color='r', linestyle='--')
            plt.plot(t1, X_MC[:, 1], color='g', linestyle='--')
            plt.plot(t1, X_MC[:, 2], color='b', linestyle='--')
            plt.plot(t1, self.Xall[0, :], color='r')
            plt.plot(t1, self.Xall[1, :], color='g')
            plt.plot(t1, self.Xall[2, :], color='b')
            plt.legend(["x", "y", "z"])
            plt.xlabel('Time [s]')
            plt.ylabel('Position [m]')
            plt.show()

        return (X_MC[:, 0].flatten(), X_MC[:, 1].flatten(), X_MC[:, 2].flatten())

if __name__=='__main__':
    run = Control_ACC()
    for i in range(3):
        if run.goal is not None:
            init_idx = run.goal
        else:
            init_idx = 1
        run.set_startgoal(init_idx)
        path, cost = run.Astar()
        print(path, cost)
        run.getPlan()
        traj = run.getSimTraj(True)