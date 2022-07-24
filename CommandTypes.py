# command types for crazyflie

class Vel:
    def __init__(self,vx=0,vy=0,vz=0):
        self.vx = vx
        self.vy = vy
        self.vz = vz
        return

class Pos:
    def __init__(self,x=0,y=0,z=0):
        self.x = x
        self.y = y
        self.z = z
        return

class Planar:
    def __init__(self,vx=0,vy=0,z=0):
        self.vx = vx
        self.vy = vy
        self.z = z
        return
