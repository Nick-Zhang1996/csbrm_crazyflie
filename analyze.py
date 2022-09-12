import pickle
import matplotlib.pyplot as plt
import numpy as np
from common import *
from time  import time,sleep
import os
import sys
from mpl_toolkits.mplot3d import Axes3D

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
#logFilename = "./logs/log1.p"
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
z_p = z-2.0
x_p = y
y_p = -x
# NOTE these are state fed to the planner
x = x_p
y = y_p
z = z_p


print_ok("actual:")
dx = np.diff(x)/(t[1]-t[0])
dy = np.diff(y)/(t[1]-t[0])
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

# ------ plot actual trajectory
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel("-x")
ax.set_ylabel("y")
ax.set_zlabel("-z")


origin = (0,-3,-5)
size = (5,5,5)
X, Y, Z = cuboid_data( origin, size )
ax.scatter(-X, Y, -Z,'k')
ax.plot(-x, y, -z, color='r', label='actual')
ax.set_xlabel('-x')
ax.set_ylabel('-y')
ax.set_zlabel('altitude (-z)')
ax.legend()
plt.show()
breakpoint()


