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
        p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
        p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])

        #Task 1
        #pick and place task for fork
        self.fork = RBOObject('fork_edit')
        self.fork.load()
        self.fork_position = [0.5, -0.3, 0.02]
        self.fork_quaternion = [0.  ,       0.    ,     0.70710678, 0.70710678]
        #Positions: Where the Fork is,Above the Center of the table, Below Prior, Back up
        self.fork_poslist = [[0.5, -0.3, 0.075],[0.55, 0.0, 0.25]]
        #Orientations: Base Orientation,Mid way to fork pose, Full Fork Pose, Base Orientation  
        #print(a)
        self.fork_quatlist = [[1.0  ,       0.    ,     0., 0.],[0.696086, 0.71789723, 0.00630872, 0.00692856]
]
        self.fork_grasp= [0,1]
        self.fork_details = {'obj':self.fork,'grasp':self.fork_grasp,'positions':self.fork_poslist,'quats':self.fork_quatlist,'num':len(self.fork_grasp)}
        self.fork.set_position_orientation(self.fork_position, self.fork_quaternion)

        #Task 2
        #Stack Cups

        # Cup 1
        self.cup1 = YCBObject("002_master_chef_can")
        self.cup1.load()
        self.cup1_position = [0.40, 0.3, 0.05]
        self.cup1_quaternion = [1.0  ,       0.    ,     0., 0.]
        #assist pos / pose
        self.cup1_poslist = [[0.40, 0.3, 0.05]]
        self.cup1_quatlist = [[1.0  ,       0.    ,     0., 0.]]
        self.cup1.set_position_orientation(self.cup1_position, self.cup1_quaternion)
        self.cup1_grasp = [0]

        self.cup1_details = {'obj':self.cup1,'grasp':self.cup1_grasp,'positions':self.cup1_poslist,'quats':self.cup1_quatlist,'num':len(self.cup1_grasp)}

        # Cup 2
        self.cup2 = YCBObject("002_master_chef_can")
        self.cup2.load()
        self.cup2_position = [0.50, 0.35, 0.05]
        self.cup2_quaternion = [1.0  ,       0.    ,     0., 0.]
        #assist pos / pose
        self.cup2_poslist = [[0.50, 0.35, 0.05]]
        self.cup2_quatlist = [[1.0  ,       0.    ,     0., 0.]]
        self.cup2_grasp = [0]
        self.cup2.set_position_orientation(self.cup2_position, self.cup2_quaternion)
        
        self.cup2_details = {'obj':self.cup2,'grasp':self.cup2_grasp,'positions':self.cup2_poslist,'quats':self.cup2_quatlist,'num':len(self.cup2_grasp)}

        # Cup 3
        self.cup3 = YCBObject("002_master_chef_can")
        self.cup3.load()
        self.cup3_position = [0.60, 0.3, 0.05]
        self.cup3_quaternion = [1.0  ,       0.    ,     0., 0.]
        #assist pos / pose
        self.cup3_poslist = [[0.55, 0.3, 0.05]]
        self.cup3_quatlist = [[1.0  ,       0.    ,     0., 0.]]
        self.cup3_grasp = [0]
        self.cup3.set_position_orientation(self.cup3_position, self.cup3_quaternion)

        self.cup3_details = {'obj':self.cup3,'grasp':self.cup3_grasp,'positions':self.cup3_poslist,'quats':self.cup3_quatlist,'num':len(self.cup3_grasp)}
                

        #Task 3
        # load mug for pose correction task
        self.mug = YCBObject("025_mug")
        self.mug.load()
        self.mug_position = [0.45, -0.1, 0.035]
        self.mug_quaternion = [ 0.5, -0.5, 0.5, -0.5 ] #horizontal
        #Positions: Where the Fork is,Above the Center of the table, Below Prior, Back up
        self.mug_poslist = [[0.45, -0.1, 0.075],[0.45, -0.15, 0.25]]
        #Orientations: Base Orientation,Mid way to fork pose, Full Fork Pose, Base Orientation  
        self.mug_quatlist = [[1.0  ,       0.    ,     0., 0.],[ 0.738987 ,  -0.00291435 ,-0.6732182 , -0.02582569]]
        self.mug_grasp = [0,1]
        self.mug.set_position_orientation(self.mug_position, self.mug_quaternion)

        self.mug_details = {'obj':self.mug,'grasp':self.mug_grasp,'positions':self.mug_poslist,'quats':self.mug_quatlist,'num':len(self.mug_grasp)}

        #Task 4
        #Load Salt + Pepper Shakers and basket

        #Salt Shaker
        self.salt = RBOObject("block")
        self.salt.load()
        self.salt_position = [0.5, 0.1, 0.05]
        self.salt_quaternion = [1.0  ,       0.    ,     0., 0.]
        #assist pos / pose
        self.salt_poslist = [[0.5, 0.1, 0.075]]
        self.salt_quatlist = [[1.0  ,       0.    ,     0., 0.]]
        self.salt_grasp = [0]
        self.salt.set_position_orientation(self.salt_position, self.salt_quaternion)

        self.salt_details = {'obj':self.salt,'grasp':self.salt_grasp,'positions':self.salt_poslist,'quats':self.salt_quatlist,'num':len(self.salt_grasp)}

        #Pepper Shaker
        self.pepper = RBOObject("block")
        self.pepper.load()
        self.pepper_position = [0.3, -0.2, 0.05]
        self.pepper_quaternion = [1.0  ,       0.    ,     0., 0.]
        #assist pos / pose
        self.pepper_poslist = [[0.3, -0.2, 0.075]]
        self.pepper_quatlist = [[1.0  ,       0.    ,     0., 0.]]
        self.pepper_grasp = [0]
        self.pepper.set_position_orientation(self.pepper_position, self.pepper_quaternion)

        self.pepper_details = {'obj':self.pepper,'grasp':self.pepper_grasp,'positions':self.pepper_poslist,'quats':self.pepper_quatlist,'num':len(self.pepper_grasp)}

        #Season Container
        self.container = RBOObject("block2")
        self.container.load()
        self.container_position = [0.65, 0.1, 0.05]
        self.container_quaternion = [ 0, 0, 0.7070727, 0.7071408 ]
        #assist pos / pose
        self.container_poslist = [[0.6, 0.1, 0.075]]
        self.container_quatlist = [[1.0  ,       0.    ,     0., 0.]]
        self.container_grasp = [0]
        self.container.set_position_orientation(self.container_position, self.container_quaternion)

        self.container_details = {'obj':self.container,'grasp':self.container_grasp,'positions':self.container_poslist,'quats':self.container_quatlist,'num':len(self.container_grasp)}


        # load a panda robot
        self.panda = Panda()
        
       # self.reset(q=[0.0, -np.pi/4, 0.0, -2*np.pi/4, 0.0, np.pi/2, 3*np.pi/4])
        #print(self.panda.state['ee_quaternion'])
        #print(self.panda.state['q'])
        #time.sleep(10)

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
