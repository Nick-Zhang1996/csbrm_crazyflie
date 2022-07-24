# interface for Optitrack Motive stream via NatNet SDK library
# Nick Zhang 2020

from NatNetClient import NatNetClient
from time import time,sleep
from threading import Event,Lock
from common import *
import numpy as np
from math import pi,radians,degrees,atan2,asin,isclose,sin,cos
#from scipy.spatial.transform import Rotation
from numeric_velocity import quad_fit_functional
from numpy import isclose


# DEBUG
from timeUtil import execution_timer


# TODO remove EKF if not needed
class Optitrack:
    def __init__(self,freq,wheelbase=102e-3):
        #self.t = execution_timer(True)
        self.newState = Event()

        # to be used in Kalman filter update
        # action = (steering in rad left positive, longitudinal acc (m/s2))
        self.action = (0,0)

        # This will create a new NatNet client
        self.streamingClient = NatNetClient()
        # set up callback functions later
        self.streamingClient.run()

        dt = 1.0/freq
        self.vel_est_n_step = 5
        self.quad_fit = quad_fit_functional(self.vel_est_n_step, dt)
        self.local_xyz_historys = []
        self.local_xyz_dots = []

        # set the relation between Optitrack world frame and our track frame
        # describe the rotation needed to rotate the world frame to track frame
        # in sequence Z, Y, X, each time using intermediate frame axis (intrinsic)
        # NOTE scipy.spatial.transform.Rotation uses active rotation
        # NOTE self.R is the passive rotation matrix
        #self.R = Rotation.from_euler("ZYX",[180,0,-90],degrees=True).inv()
        # custome routine is faster
        # quicknote: in mk425
        # x pointing away from door
        # y point to right
        # z downward
        self.R = self.eulerZyxToR(radians(180),0,radians(-90))

        # with world ref frame, where is local frame origin
        self.local_frame_origin_world = np.array([0,0,0])

        # to convert a vector in world frame to local/track frame
        # vector_world_frame -= local_frame_origin_world
        # vector_track_frame = self.R.apply(vector_world_frame)


        # a list of state tuples, state tuples take the form: (x,y,z,qx,qy,qz,qw), in meters and unit quaternion
        self.state_list = []
        # note that rx,ry,rz are euler angles in ZYX convention as commonly used in aviation
        # local state (x,y,z,rx,ry,rz)
        self.local_state_list = []
        self.kf_state_list = []
        self.lost = []
        self.state_lock = Lock()

        # a mapping from internal id to optitrack id
        # self.optitrack_id_lookup[internal_id] = optitrack_id
        self.optitrack_id_lookup = []

        self.obj_count = 0

        # set callback for rigid body state update, this will create a new KF instance for each object
        # and set up self.optitrack_id_lookup table
        self.streamingClient.rigidBodyListener = self.receiveRigidBodyFrameInit
        # wait for all objects to be detected
        sleep(0.1)
        # switch to regular callback now that everything is initialized
        self.streamingClient.rigidBodyListener = self.receiveRigidBodyFrame



    def __del__(self):
        self.streamingClient.requestQuit()

    def quit(self):
        self.streamingClient.requestQuit()
        #self.t.summary()

    # there are two sets of id
    # Optitrack ID: like object name in vicon, each object has a unique ID that can be any integer value
    # internal ID within this class, like object id in vicon, each object has a unique id, id will be assigned starting from zero
    # for example, the Optitrack ID for two objects may be 7,9, while their corresponding internal ID will be 0,1
    # this is to facilitate easier indexing 
    def getOptitrackId(self,internal_id):
        # hard code since we only have a handful of models
        try:
            return self.optitrack_id_lookup[internal_id]
        except IndexError:
            print_error("can't find internal ID %d"%internal_id)
            return None

    # find internal id from optitrack id
    def getInternalId(self,optitrack_id):
        try:
            return self.optitrack_id_lookup.index(optitrack_id)
        except ValueError:
            print_error("can't find optitrack ID %d"%optitrack_id)
            return None

    # optitrack callback for item discovery
    # this differs from receiveRigidBodyFrame in that
    # 1. does not include kalman filter update
    # 2. if an unseen id is found, it will be added to id list and an KF instance will be created for it
    def receiveRigidBodyFrameInit( self, optitrack_id, position, rotation ):

        if not (optitrack_id in self.optitrack_id_lookup):
            self.obj_count +=1
            self.optitrack_id_lookup.append(optitrack_id)

            x,y,z = position
            qx, qy, qz, qw = rotation
            r = self.quatToR(qx,qy,qz,qw)
            # get body pose in track frame
            # x,y,z in track frame
            
            x_local, y_local, z_local = self.R @ (np.array(position) - self.local_frame_origin_world)

            local_r = r @ np.linalg.inv(self.R)
            rx_local, ry_local, rz_local = self.rotationToEulerZyx(local_r)
            
            self.lost.append(Event())

            self.state_lock.acquire(timeout=0.01)
            self.state_list.append((x,y,z,qx,qy,qz,qw))
            self.local_state_list.append((x_local,y_local,z_local,rx_local,ry_local,rz_local))
            self.local_xyz_historys.append([])
            for i in range(self.vel_est_n_step-1):
                self.local_xyz_historys[-1].append([x_local,y_local,z_local])

            self.local_xyz_dots.append([0,0,0])
            self.state_lock.release()

    # regular callback for state update
    def receiveRigidBodyFrame(self, optitrack_id, position, rotation ):
        #self.t.s()
        #print( "Received frame for rigid body", id )

        #self.t.s('internal id lookup')
        internal_id = self.getInternalId(optitrack_id)
        x,y,z = position
        qx, qy, qz, qw = rotation
        #self.t.e('internal id lookup')

        # detect if object is lost
        # TODO verify this works
        #self.t.s('lose track guard')
        s = self.state_list[internal_id]
        old_x = s[0]
        old_y = s[1]
        old_z = s[2]
        if (x==old_x and y==old_y and z==old_z):
            #self.lost[internal_id].set()
            pass
        #self.t.e('lose track guard')

        # r is active rotation
        #self.t.s('ref1 alt')
        r = self.quatToR(qx,qy,qz,qw)
        #self.t.e('ref1 alt')
        # euler angle in world ref frame is not used by client, calculate only by request in getState()
        #rz, ry, rx = r.as_euler('ZYX',degrees=False)

        # get body pose in local frame
        #self.t.s('ref3 alt')
        x_local, y_local, z_local = self.R @ (np.array(position) - self.local_frame_origin_world)
        #self.t.e('ref3 alt')

        #self.t.s('ref4')
        #local_r = self.R*r
        local_r = r @ np.linalg.inv(self.R)
        #self.t.e('ref4')

        #self.t.s('ref5')
        # use custom function for faster calculation
        # rotationToEulerZyx is about 6x faster than as_euler()
        #rz_local, ry_local, rx_local = local_r.as_euler("ZYX",degrees=False)
        rx_local, ry_local, rz_local = self.rotationToEulerZyx(local_r)
        #self.t.e('ref5')

        #self.t.s('velocity')
        self.local_xyz_historys[internal_id].append([x_local,y_local,z_local])
        temp_xyz_history = np.array(self.local_xyz_historys[internal_id])
        x_hist = temp_xyz_history[:,0]
        y_hist = temp_xyz_history[:,1]
        z_hist = temp_xyz_history[:,2]
        vx = self.quad_fit(x_hist)
        vy = self.quad_fit(y_hist)
        vz = self.quad_fit(z_hist)
        self.local_xyz_dots[internal_id] = [vx,vy,vz]

        self.local_xyz_historys[internal_id].pop(0)
        #self.t.e('velocity')


        #self.t.s('update')
        self.state_lock.acquire(timeout=0.01)
        self.state_list[internal_id] = (x,y,z,qx,qy,qz,qw)
        self.local_state_list[internal_id] = (x_local,y_local,z_local,rx_local,ry_local,rz_local)

        self.state_lock.release()
        self.newState.set()
        #self.t.e('update')
        #print("Internal ID: %d \n Optitrack ID: %d"%(i,op_id))
        #print("World coordinate: %0.2f,%0.2f,%0.2f"%(x,y,z))
        #print("local state: %0.2f,%0.2f, heading= %0.2f"%(x_local,y_local,theta_local))
        #print("\n")
        #print(1.0/(time()-tic))
        #self.t.e()
        return


    # r is passive rotation matrix
    def rotationToEulerZyx(self,r):
        #r = r.inv().as_matrix()
        rx = atan2(r[1,2],r[2,2])
        ry = -asin(r[0,2])
        rz = atan2(r[0,1],r[0,0])
        return (rx,ry,rz)

    # get the passive rotation matrix 
    # that is, x_body = R*x_world
    def eulerZyxToR(self,rz,ry,rx):
        R = [ [ cos(ry)*cos(rz), cos(ry)*sin(rz), -sin(ry)],
              [ sin(rx)*sin(ry)*cos(rz)-cos(rx)*sin(rz), sin(rx)*sin(ry)*sin(rz)+cos(rx)*cos(rz), cos(ry)*sin(rx)],
              [cos(rx)*sin(ry)*cos(rz)+sin(rx)*sin(rz), cos(rx)*sin(ry)*sin(rz)-sin(rx)*cos(rz), cos(ry)*cos(rx)]]
        
        return R

    # get passive rotation matrix
    def quatToR(self,q1,q2,q3,q0):
        # q1=qx
        # q2=qy
        # q3=qz
        # q0=q2
        R = [ [q0**2+q1**2-q2**2-q3**2, 2*q1*q2+2*q0*q3, 2*q1*q3-2*q0*q2],
              [2*q1*q2-2*q0*q3,  q0**2-q1**2+q2**2-q3**2, 2*q2*q3+2*q0*q1],
              [2*q1*q3+2*q0*q2,  2*q2*q3-2*q0*q1, q0**2-q1**2-q2**2+q3**2]]
        
        return R

    # self.R*x_world
    def worldToLocal(self,world):
        return x,y,z


    def getLocalVelocity(self,internal_id):
        return self.local_xyz_dots[internal_id]

    def getLocalState(self,internal_id):
        if internal_id>=self.obj_count:
            print_error("can't find internal id %d"%(internal_id))
            return None
        self.state_lock.acquire(timeout=0.01)
        retval = self.local_state_list[internal_id]
        self.state_lock.release()
        return retval
    

    # get Global(optitrack world frame)state by internal id
    def getState(self, internal_id):
        if internal_id>=self.obj_count:
            print_error("can't find internal id %d"%(internal_id))
            return None
        # calculate rx,ry,rz from quaternion
        x,y,z, qx,qy,qz,qw = self.state_list[internal_id]
        r = self.quatToR(qx,qy,qz,qw)
        rx, ry, rz = self.rotationToEulerZyx(r)
        self.state_lock.acquire(timeout=0.01)
        retval = (x,y,z,rx,ry,rz)
        self.state_lock.release()
        return retval

# test functionality
if __name__ == '__main__':
    op = Optitrack(freq=50)
    #for i in range(op.obj_count):
    op_id = 6
    i = op.getInternalId(op_id)
    print("Internal ID: %d \n Optitrack ID: %d"%(i,op_id))
    while True:
        #op_id = op.getOptitrackId(i)
        #x,y,z,rx,ry,rz = op.getState(i)
        #print("World coordinate: %0.2f,%0.2f,%0.2f"%(x,y,z))
        #print("rx: %0.2f, ry: %0.2f, rz: %0.2f"%(degrees(rx),degrees(ry),degrees(rz)))

        x_local, y_local, z_local, rx_local, ry_local, rz_local = op.getLocalState(i)
        print("local state: %0.2f,%0.2f, %0.2f, heading= %0.2f"%(x_local,y_local,z_local,degrees(rz_local)))
        print("\n")
        sleep(0.03)

    input("press enter to stop\n")
    op.quit()


