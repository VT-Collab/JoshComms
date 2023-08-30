import os
import numpy as np
import pybullet as p
import pybullet_data
from panda_env import Panda
from objects import RBOObject
import pickle
import tf
import time
from Utils import *
# from objects import YCBObject, InteractiveObj, RBOObject


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
        #p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, -1, -0.65])
        p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
        #p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, +1, -0.65])

        #for goal in range (len(goals)):
            #p.loadURDF(os.path.join(self.urdfRootPath, "sphere_small.urdf"), basePosition=goals[goal], useFixedBase=1)

        # load obstacle (soccerball)
        # p.loadURDF(os.path.join(self.urdfRootPath, "soccerball.urdf"), basePosition=[0.6, 0.1 - 0.6, 0.07], globalScaling=0.20)
        
        # load block1
        self.block1 = RBOObject('block')
        self.block1.load()
        self.block1_position = [0.5, -0.35, 0.05]
        self.block1_quaternion = [0, 0, 0, 1]
        self.block1.set_position_orientation(self.block1_position, self.block1_quaternion)
        #load block1
        self.block2 = RBOObject('block')
        self.block2.load()
        self.block2_position = [0.5, -0.4, 0.05]
        self.block2_quaternion = [0, 0, 0, 1]
        self.block2.set_position_orientation(self.block2_position, self.block2_quaternion)
        # load block3
        self.block3 = RBOObject('block')
        self.block3.load()
        self.block3_position = [0.5, -0.3, 0.05]
        self.block3_quaternion = [0, 0, 0, 1]
        self.block3.set_position_orientation(self.block3_position, self.block3_quaternion)

        # # load door
        # self.door = RBOObject('door')
        # self.door.load()
        # self.door_position = [0.3, -0.7, -0.4]
        # self.door_quaternion = [0.70710678, 0.   ,      0.    ,     0.70710678]
        # #print("ABAAAAAAAAAAAAAAAAAAAAAAAAAAAA",tf.quaternion_from_euler(np.pi/2, 0, 0, axes='sxyz'))
        # self.door_angle = -0.5
        # self.door.set_position_orientation(self.door_position, self.door_quaternion)
        # p.resetJointState(self.door.body_id, 1, self.door_angle)

        # # load block1
        # self.box = RBOObject('cabinet')
        # self.box.load()
        # self.box_position = [0.5, 0.2, 0.0]
        # self.box_quaternion = [0, 0, 0, 1]
        # self.box.set_position_orientation(self.box, self.box)


        # load a panda robot
        self.panda = Panda()
    
    def reset_box(self):
        self.block.set_position_orientation(self.block_position, self.block_quaternion)
    
    def reset(self, q=[0.0, -np.pi/4, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4]):
        self.panda.reset(q)
        return [self.panda.state]

    def close(self):
        p.disconnect()

    def step(self, action):
        # get current state
        state = self.panda.state['q']
        
        goal =  state[:7] + action

        pos1,pose = joint2pose(goal)
        pos2,pose = joint2pose(action)
        # print("Goal Position", pos1)
        # print("Goal Position", pos2)
        # print("State", state)
        # print("action", action)
        # print("goal", goal)
        while (np.linalg.norm(goal-state[:7]) > .5):
            state = self.panda.state['q']
            act2 =(goal-state[:7])
            self.panda.traj_q(qd=goal)
            print("WE LIKE TO MOVE IT:",np.linalg.norm(goal-state[:7]))
            print(act2)
            print(state[:7])
            # take simulation step
            p.stepSimulation()
            time.sleep(.05)
            #print(state)

        # return next_state, reward, done, info
        next_state = [self.panda.state]
        reward = 0.0
        done = False
        info = {}
        return next_state, reward, done, info

    # # input trajectory, output final box position
    # def play_traj(self, xi, T=2.0):
    #     traj = Trajectory(xi, T)
    #     self.panda.reset_task(xi[0, :], [1, 0, 0, 0])
    #     print(self.panda.read_state())
    #     self.reset_box()
    #     sim_time = 0
    #     while sim_time < T:
    #         self.panda.traj_task(traj, sim_time)
    #         p.stepSimulation()
    #         sim_time += 1/240.0 # this is the default step time in pybullet
    #         # time.sleep(1/240.0) # for real-time visualization
    #     return self.read_box()
    
    def state(self):
        return self.panda.state

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
        p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=90, cameraPitch=-31.4,
                                     cameraTargetPosition=[1.1, 0.0, 0.0])
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
