import os
import numpy as np
import pybullet as p
import pybullet_data
from panda_env2 import Panda
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
        p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
        p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])

        # # load block1
        # self.block1 = RBOObject('block')
        # self.block1.load()
        # self.block1_position = [0.5, -0.35, 0.05]
        # self.block1_quaternion = [0, 0, 0, 1]
        # self.block1.set_position_orientation(self.block1_position, self.block1_quaternion)
        # #load block1
        # self.block2 = RBOObject('block')
        # self.block2.load()
        # self.block2_position = [0.5, -0.4, 0.05]
        # self.block2_quaternion = [0, 0, 0, 1]
        # self.block2.set_position_orientation(self.block2_position, self.block2_quaternion)
        # # load block3
        # self.block3 = RBOObject('block')
        # self.block3.load()
        # self.block3_position = [0.5, -0.3, 0.05]
        # self.block3_quaternion = [0, 0, 0, 1]
        # self.block3.set_position_orientation(self.block3_position, self.block3_quaternion)

        # # load door
        # self.door = RBOObject('door')
        # self.door.load()
        # self.door_position = [0.3, -0.7, -0.4]
        # self.door_quaternion = [0.70710678, 0.   ,      0.    ,     0.70710678]
        # #print("ABAAAAAAAAAAAAAAAAAAAAAAAAAAAA",tf.quaternion_from_euler(np.pi/2, 0, 0, axes='sxyz'))
        # self.door_angle = -0.5
        # self.door.set_position_orientation(self.door_position, self.door_quaternion)
        # p.resetJointState(self.door.body_id, 1, self.door_angle)

        # load block1
        # fork in food = [ 0.74968156 , 0.00226022 ,-0.65986552 ,-0.05049703],[.4,0,.4]
        #positions, Pick up fork, move fork to center of table where food is, move fork to stabbing position, reverse to normal
        #print("ABAAAAAAAAAAAAAAAAAAAAAAAAAAAA",quaternion_from_matrix([-0.0545892,0.238157,-0.0895843,-1.89678,-0.0612414,1.11555,0.693995]))
        self.fork = RBOObject('fork_edit')
        self.fork.load()
        self.fork_position = [0.5, 0.3, 0.02]
        
        self.fork_quaternion = [0.  ,       0.    ,     0.70710678, 0.70710678]
        #Positions: Where the Fork is,Above the Center of the table, Below Prior, Back up
        self.fork_poslist = [[0.5, 0.3, 0.02],[0.4, 0.0, 0.03],[0.4, 0.0, 0.02],[0.5, 0.-0.3, 0.2]]
        #Orientations: Base Orientation,Mid way to fork pose, Full Fork Pose, Base Orientation  
        self.fork_quatlist = [[1.0  ,       0.    ,     0., 0.],[ 0.87197471, -0.02540665, -0.48859053, -0.01714349],[ 0.74968156 , 0.00226022 ,-0.65986552 ,-0.05049703],[1.0  ,       0.    ,     0. , 0.]]
        self.fork_grasp= [0,1,1,1]
        self.fork_details = {'obj':self.fork,'grasp':self.fork_grasp,'positions':self.fork_poslist,'quats':self.fork_quatlist,'num':len(self.fork_grasp)}
        self.fork.set_position_orientation(self.fork_position, self.fork_quaternion)
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
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=30, cameraPitch=-60,
                                     cameraTargetPosition=[0.5, -0.2, 0.0])
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
