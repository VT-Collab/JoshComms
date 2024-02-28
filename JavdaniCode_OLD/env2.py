import os
import numpy as np
import pybullet as p
import pybullet_data
from panda_env import Panda
from objects import YCBObject, InteractiveObj, RBOObject
from tf import *
import time
# my comment!

class SimpleEnv():

    def __init__(self, visualize=False):
        # create simulation (GUI)
        #goals = pickle.load(open("goals/goals" + str(env_goals) + ".pkl", "rb"))
        self.urdfRootPath = pybullet_data.getDataPath()
        if visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT) 
        p.setGravity(0, 0, -9.81)
        # set up camera
        self._set_camera()
        # load some scene objects
        p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[-.5, 0, -0.65])
        p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[-0.25, 0, -0.65])

        self.init_objects()

        # load a panda robot
        self.panda = Panda()


    def reset_box(self):
        self.block.set_position_orientation(self.block_position, self.block_quaternion)
    
    def reset(self, q=[0.0, -np.pi/4, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4]):
        self.panda.reset(q)
        return [self.panda.state]

    def close(self):
        p.disconnect()

    def state(self):
        return self.panda.state


    def step(self, joint =[0]*7,pos=[0]*3,quat =[0]*4 ,grasp = True,mode = 1):
        # get current state djoint=[0]*7, dposition=[0]*3, dquaternion=[0]*4
        state = self.panda.state
        #print(pos)

        self.panda.step(mode=mode,djoint=joint,dposition=pos,dquaternion=quat,grasp_open=grasp)

        # take simulation step
        p.stepSimulation()
        time.sleep(.01)
        
        # return next_state, reward, done, info
        next_state = self.panda.state
        #diff = next_state['q'] - state['q']
        #print(action)
        #print(diff)
        reward = 0.0
        done = False
        info = {}
        return next_state, reward, done, info

    def render(self):
        (width, height, pxl, depth, segmentation) = p.getCameraImage(width=self.camera_width,
                                                                     height=self.camera_height,
                                                                     viewMatrix=self.view_matrix,
                                                                     projectionMatrix=self.proj_matrix)
        rgb_array = np.array(pxl, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self.camera_height, self.camera_width, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _set_camera(self):
        self.camera_width = 256
        self.camera_height = 256
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=-30, cameraPitch=-60,
                                     cameraTargetPosition=[-0.5, -0.2, 0.0])
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0, 0],
                                                               distance=1.0,
                                                               yaw=90,
                                                               pitch=-50,
                                                               roll=0,
                                                               upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(self.camera_width) / self.camera_height,
                                                        nearVal=0.1,
                                                        farVal=100.0)
    def init_objects(self):

        # #Task A
        # #Load Salt + Pepper Shakers 

        #Salt Shaker
        self.salt = RBOObject("block")
        self.salt.load()
        self.salt_position = [-0.38, 0.105, 0.25]
        self.salt_poslist = [[-0.38, 0.105, 0.25]]
        self.salt_quaternion = [1.0  ,       0.    ,     0., 0.]
        #assist pos / pose
        
        self.salt_quatlist = [[1.0  ,       0.    ,     0., 0.]]
        self.salt_grasp = [0]
        self.salt.set_position_orientation(self.salt_position, self.salt_quaternion)
        self.salt_name = ["Salt"]

        self.salt_details = {'obj':self.salt,'grasp':self.salt_grasp,'name': self.salt_name,'positions':self.salt_poslist,'quats':self.salt_quatlist,'num':len(self.salt_grasp)}

        #Salt Shaker
        self.pepper = RBOObject("block")
        self.pepper.load()
        self.pepper_position = [-0.78, -0.016, 0.3]
        self.pepper_poslist = [[-0.78, -0.016, 0.3]]
        self.pepper_quaternion = [1.0  ,       0.    ,     0., 0.]
        #assist pos / pose
        
        self.pepper_quatlist = [[1.0  ,       0.    ,     0., 0.]]
        self.pepper_grasp = [0]
        self.pepper.set_position_orientation(self.pepper_position, self.pepper_quaternion)
        self.pepper_name = ["Pepper"]

        self.pepper_details = {'obj':self.pepper,'grasp':self.pepper_grasp,'name': self.pepper_name,'positions':self.pepper_poslist,'quats':self.pepper_quatlist,'num':len(self.pepper_grasp)}

        self.plate = RBOObject("plate")
        self.plate.load()
        self.plate_position = [-0.555, -0.265, 0.3]
        self.plate_poslist = [[-0.555, -0.265, 0.3]]
        self.plate_quaternion = [1.0  ,       0.    ,     0., 0.]
        #assist pos / pose
        
        self.plate_quatlist = [[1.0  ,       0.    ,     0., 0.]]
        self.plate_grasp = [0]
        self.plate.set_position_orientation(self.plate_position, self.plate_quaternion)
        self.plate_name = ["plate"]

        self.plate_details = {'obj':self.plate,'grasp':self.plate_grasp,'name': self.plate_name,'positions':self.plate_poslist,'quats':self.plate_quatlist,'num':len(self.plate_grasp)}

        #Task B
        #Load Fork + Spoon + knife + tray

        #Fork Spawn
        self.fork = RBOObject("block")
        self.fork.load()
        self.fork_position = [-0.59, 0.22, 0.3]
        self.fork_poslist = [[-0.59, 0.22, 0.3]]
        self.fork_quaternion = [1.0  ,       0.    ,     0., 0.]
        #assist pos / pose
        
        self.fork_quatlist = [[1.0  ,       0.    ,     0., 0.]]
        self.fork_grasp = [0]
        self.fork.set_position_orientation(self.fork_position, self.fork_quaternion)
        self.fork_name = ["Fork"]

        self.fork_details = {'obj':self.fork,'grasp':self.fork_grasp,'name': self.fork_name,'positions':self.fork_poslist,'quats':self.fork_quatlist,'num':len(self.fork_grasp)}

        #Spoon Spawn
        self.spoon= RBOObject("block")
        self.spoon.load()
        self.spoon_position = [-.6, 0.088, 0.3]
        self.spoon_poslist = [[-0.6, 0.088, 0.3]]
        self.spoon_quaternion = [1.0  ,       0.    ,     0., 0.]
        #assist pos / pose
        
        self.spoon_quatlist = [[1.0  ,       0.    ,     0., 0.]]
        self.spoon_grasp = [0]
        self.spoon.set_position_orientation(self.spoon_position, self.spoon_quaternion)
        self.spoon_name = ["Spoon"]

        self.spoon_details = {'obj':self.spoon,'grasp':self.spoon_grasp,'name': self.spoon_name,'positions':self.spoon_poslist,'quats':self.spoon_quatlist,'num':len(self.spoon_grasp)}

        #Fork Goal
        self.fork2 = RBOObject("block")
        self.fork2.load()
        self.fork2_position = [-0.36, -0.265, 0.3]
        self.fork2_poslist = [[-0.36, -0.265, 0.3]]
        self.fork2_quaternion = [1.0  ,       0.    ,     0., 0.]
        #assist pos / pose
        
        self.fork2_quatlist = [[1.0  ,       0.    ,     0., 0.]]
        self.fork2_grasp = [0]
        self.fork2.set_position_orientation(self.fork2_position, self.fork2_quaternion)
        self.fork2_name = ["Fork2"]

        self.fork2_details = {'obj':self.fork2,'grasp':self.fork2_grasp,'name': self.fork2_name,'positions':self.fork2_poslist,'quats':self.fork2_quatlist,'num':len(self.fork2_grasp)}

        #Spoon Goal
        self.spoon2= RBOObject("block")
        self.spoon2.load()
        self.spoon2_position = [-0.755, -0.3, 0.3]
        self.spoon2_poslist = [[-0.755, -0.3, 0.3]]
        self.spoon2_quaternion = [1.0  ,       0.    ,     0., 0.]
        #assist pos / pose
        
        self.spoon2_quatlist = [[1.0  ,       0.    ,     0., 0.]]
        self.spoon2_grasp = [0]
        self.spoon2.set_position_orientation(self.spoon2_position, self.spoon2_quaternion)
        self.spoon2_name = ["Spoon2"]

        self.spoon2_details = {'obj':self.spoon2,'grasp':self.spoon2_grasp,'name': self.spoon2_name,'positions':self.spoon2_poslist,'quats':self.spoon2_quatlist,'num':len(self.spoon2_grasp)}

        # #Knife 
        # self.knife= RBOObject("block")
        # self.knife.load()
        # self.knife_position = [0.4, -0.3, 0.1]
        # self.knife_poslist = [[0.4, -0.3, 0.1]]
        # self.knife_quaternion = [1.0  ,       0.    ,     0., 0.]
        # #assist pos / pose
        
        # self.knife_quatlist = [[1.0  ,       0.    ,     0., 0.]]
        # self.knife_grasp = [0]
        # self.knife.set_position_orientation(self.knife_position, self.knife_quaternion)
        # self.knife_name = ["Spoon"]

        # self.knife_details = {'obj':self.knife,'grasp':self.knife_grasp,'name': self.knife_name,'positions':self.knife_poslist,'quats':self.knife_quatlist,'num':len(self.knife_grasp)}

        # #Tray
        # self.tray= RBOObject("tray")
        # self.tray.load()
        # self.tray_position = [0.5, 0.3, 0.1]
        # self.tray_poslist = [[0.5, 0.3, 0.1]]
        # self.tray_quaternion = [1.0  ,       0.    ,     0., 0.]
        # #assist pos / pose
        
        # self.tray_quatlist = [[1.0  ,       0.    ,     0., 0.]]
        # self.tray_grasp = [0]
        # self.tray.set_position_orientation(self.tray_position, self.tray_quaternion)
        # self.tray_name = ["Tray"]

        # self.tray_details = {'obj':self.tray,'grasp':self.tray_grasp,'name': self.tray_name,'positions':self.tray_poslist,'quats':self.tray_quatlist,'num':len(self.tray_grasp)}
       
       
        #Task C -----------------------------------------------
        # Cup 1
        self.cup1 = YCBObject("002_master_chef_can")
        self.cup1.load()
        self.cup1_position = [-0.285, -0.065, 0.3]
        self.cup1_poslist = [[-0.285, -0.065, 0.3]]
        self.cup1_quaternion = [1.0  ,       0.    ,     0., 0.]
        #assist pos / pose
        
        self.cup1_quatlist = [[1.0  ,       0.    ,     0., 0.]]
        self.cup1.set_position_orientation(self.cup1_position, self.cup1_quaternion)
        self.cup1_grasp = [0]
        self.cup1_name = ["Can"]
        self.cup1_details = {'obj':self.cup1,'grasp':self.cup1_grasp,'name': self.cup1_name,'positions':self.cup1_poslist,'quats':self.cup1_quatlist,'num':len(self.cup1_grasp)}

        
        # load mug for pose correction task
        self.mug = YCBObject("025_mug")
        self.mug.load()
        self.mug_position = [-0.68, -0.08, 0.3]
        self.mug_poslist = [[-0.68, -0.08, 0.3]]
        self.mug_quaternion = [ 0.5, -0.5, 0.5, -0.5 ] #horizontal
        self.mug.set_position_orientation(self.mug_position, self.mug_quaternion)
        #Positions: Where the Fork is,Above the Center of the table, Below Prior, Back up
        
        #Orientations: Base Orientation,Mid way to fork pose, Full Fork Pose, Base Orientation  
        self.mug_quatlist = [[1.0  ,       0.    ,     0., 0.]]
        self.mug_grasp = [0]
        
        self.mug_name = ["Mug"]
        self.mug_details = {'obj':self.mug,'grasp':self.mug_grasp,'name': self.mug_name,'positions':self.mug_poslist,'quats':self.mug_quatlist,'num':len(self.mug_grasp)}
