# analyze log, specifically for csbrm planner
import pickle
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from common import *
from time  import time,sleep
import os
import sys
from mpl_toolkits.mplot3d import Axes3D

#from csbrmDemo import Control_ACC,getSimTraj
from CSBRMlanding import Control_ACC
csbrm = Control_ACC()
# NOTE temp
init_idx = 1
csbrm.set_startgoal(init_idx)
path, cost = csbrm.Astar()
print(path, cost)
csbrm.getPlan()
x_des,y_des,z_des = csbrm.getSimTraj()

def cuboid_data(o, size=(1,1,1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1], o[1], o[1]],
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
    z = [[o[2], o[2], o[2], o[2], o[2]],
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]
    return np.array(x), np.array(y), np.array(z)


# ------  Load data ------

output = open('log.p','rb')
data = pickle.load(output)
output.close()

data = np.array(data)
print("log length %.1f seconds"%(data.shape[0]/119.88))

skip = 0
t = data[skip:,0] - data[0,0]
x = data[skip:,1]
y = data[skip:,2]
z = data[skip:,3]
rx = data[skip:,4]
ry = data[skip:,5]
rz = data[skip:,6]

# TODO should this be vx?
accx = data[skip:,7]
accy = data[skip:,8]
accz = data[skip:,9]

# convert vicon frame to planner frame
offset = 0
x_p = x
y_p = -y
z_p = -z+offset
x = x_p
y = y_p
z = z_p

print_ok('1/dt=',1/(t[1]-t[0]))

dt = 0.005
# dt = t[1]-t[0]

print_ok("actual:")
dx = np.diff(x)/dt
dy = np.diff(y)/dt
dz = np.diff(z)/dt
ddx = np.diff(dx)/dt
ddy = np.diff(dy)/dt
max_speed = np.max((dx*dx + dy*dy))**0.5
max_acc = np.max((ddx*ddx + ddy*ddy))**0.5
print_info("trajectory time %.2f" %(t[-1]))
print_info("(?)max speed : %.1f m/s "%(max_speed))
print_info("(?)max acc : %.1f m/s "%(max_acc))
'''
print_info("speed")
plt.plot((dx*dx + dy*dy)**0.5)
plt.show()
print_info("acc")
plt.plot((ddx*ddx + ddy*ddy)**0.5,'o')
plt.show()
print_info("roll/pitch")
plt.plot(rx*180.0/np.pi)
plt.plot(ry*180.0/np.pi)
plt.show()
'''
# ---- reconstruct planner response -----
vx = dx
vy = dy
vz = dz

acc_norm_vec = []
acc_planner = np.array([[0], [0], [0]])
last_ts = -1
for i in range(t.shape[0]-1):
    state_planner = (x[i], y[i], z[i], vx[i], vy[i], vz[i])
    time_step = int(t[i] * 50)
    acc_des_planner = csbrm.MCplan(np.array(state_planner), time_step)
    if (len(acc_des_planner) == 0):
        break
    acc_planner = np.append(acc_planner, acc_des_planner, axis=1)
    acc_des_norm = np.linalg.norm(acc_des_planner)
    acc_norm_vec.append(acc_des_norm)
acc_planner = np.delete(acc_planner, 0, 1)

plt.plot(t[:-1],acc_norm_vec)
plt.xlabel('time(s)')
plt.ylabel('Requested acceleration (m/s2)')
plt.title('Requested acceleration (m/s2)')
plt.show()


# ------ plot actual trajectory
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

#x_des,y_des,z_des = getSimTraj()

origin = (0,-3,-5)
size = (5,5,5)
X, Y, Z = cuboid_data( origin, size )
ax.scatter(X, Y, Z,'k')
ax.plot(x, y, z, color='r', label='actual')
ax.plot(x_des, y_des, z_des, color='b', label='expected')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()

t_des = 1/120 * np.arange(len(x_des))

# scipy.io.savemat('./Matlabdata/ref.mat', mdict={'t_des': t_des, 'x_des': x_des, 'y_des': y_des, 'z_des': z_des})

scipy.io.savemat('./Matlabdata/final' + text + '.mat', mdict={'t': t, 'x_p': x_p, 'y_p': y_p, 'z_p': z_p,
                                                              'x_v': dx, 'y_v': dy, 'z_v': dz,
                                                              'x_a': accx, 'y_a': accy, 'z_a': accz,
                                                              'acc_compute': acc_planner})

plt.title('Expected vs Actual position')
plt.plot(t,x,'-', color='r')
plt.plot(t_des,x_des,'--', color='r')
plt.plot(t,y,'-', color='b')
plt.plot(t_des,y_des,'--', color='b')
plt.plot(t,z,'-', color='g')
plt.plot(t_des,z_des,'--', color='g')
plt.xlabel('time(s)')

plt.show()
