# analyze log, specifically for csbrm planner
import pickle
import matplotlib.pyplot as plt
import numpy as np
from common import *
from time  import time,sleep
import os
import sys
from mpl_toolkits.mplot3d import Axes3D

from csbrmDemo import Control_ACC,getSimTraj
csbrm = Control_ACC()

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
logFilename = "./log.p"
output = open(logFilename,'rb')
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
# convert vicon frame to planner frame
x_p = x
y_p = -y
z_p = -z+1.2
x = x_p
y = y_p
z = z_p
print_ok('1/dt=',1/(t[1]-t[0]))


print_ok("actual:")
dx = np.diff(x)/(t[1]-t[0])
dy = np.diff(y)/(t[1]-t[0])
dz = np.diff(z)/(t[1]-t[0])
ddx = np.diff(dx)/(t[1]-t[0])
ddy = np.diff(dy)/(t[1]-t[0])
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
'''
acc_norm_vec = []
last_ts = -1
for i in range(t.shape[0]-1):
    state_planner = (y[i], -x[i], -z[i], vy[i], -vx[i], -vz[i])
    #time_step = int(t[i] / 0.1)
    time_step = int(t[i] * 120)
    acc_des_planner = csbrm.MCplan(np.array(state_planner), time_step)
    acc_des_norm = np.linalg.norm(acc_des_planner)
    acc_norm_vec.append(acc_des_norm)
plt.plot(t[:-1],acc_norm_vec)
plt.xlabel('time(s)')
plt.ylabel('Requested acceleration (m/s2)')
plt.title('Requested acceleration (m/s2)')
plt.show()
'''

# ------ plot actual trajectory
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

x_des,y_des,z_des = getSimTraj()

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

t_des = 1/10 * np.arange(len(x_des))
plt.title('Expected vs Actual position')
plt.plot(t,x,'-', color='r')
plt.plot(t_des,x_des,'--', color='r')
plt.plot(t,y,'-', color='b')
plt.plot(t_des,y_des,'--', color='b')
plt.plot(t,z,'-', color='g')
plt.plot(t_des,z_des,'--', color='g')
plt.xlabel('time(s)')

plt.show()


