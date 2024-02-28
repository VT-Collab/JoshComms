#The basic ADA interface, such as executing end effector motions, getting the state of the robot, etc.
from Primary_AssistPolicy import *
from UserBot import *
#import AssistancePolicyVisualizationTools as vistools
from Utils import *
#from GoalPredictor import *
from GUI import *

#from DataRecordingUtils import TrajectoryData
import pandabase as panda
import numpy as np
import math
import time
import os, sys
#import cPickle as pickle
import argparse
import copy
#import tf
import tf as transmethods
#import rospkg
#import rospy
#import rosli


rospy.init_node("teleop")

mover = TrajectoryClient()
joystick = Joystick()
start_time = timestep = timest = time.time()
rate = rospy.Rate(1000)
motor = False
pick = False
pick_count = 0

print("[*] Initialized, Moving Home")
mover.switch_controller(mode='position')
mover.send_joint(HOME, 2.0)
mover.client.wait_for_result()
mover.switch_controller(mode='velocity')
print("[*] Ready for joystick inputs")

q = mover.samplejoint()
goal = np.array([-0.28, 0.102, 0.25])
while not rospy.is_shutdown():
    x = mover.joint2pose()
    pose = mover.dirkin(q)
    #print((goal),pose[:3,3])
    pose[:3,3] = (3*(goal)+pose[:3,3])/4
    goal_q = mover.invkin(pose,q)
    qdot =  (goal_q - q) 
    qdot *= .1/(np.linalg.norm(qdot))
    print("TESTIN",(np.array(qdot)))
    mover.sendQ(np.reshape(np.array(qdot),(1,6)))
    mover.client.wait_for_result()
    if

def get_action(self,q=None,goal_pos=None ):
    if goal_pos == None:
        goal_pos = self.goal_pos
    if q == None:
        q = self.robot_state["q"]
    #goal_euler= np.array(transmethods.euler_from_quaternion(self.goal_quat))
    #goal_x = np.append(self.goal_pos,[0,0,0])
    pose = self.kdl_kin.forward(q)
    #print(np.reshape(goal_pos,(3,1)),pose[:3,3])
    pose[:3,3] = (np.reshape(goal_pos,(3,1))+pose[:3,3])/2
    #print(pose)
    goal_q = self.invkin(pose)
    qdot =  (goal_q - q) 
    print(goal_q)
    qdot *= .1/(np.linalg.norm(qdot))

    #print("SHAPIN UP",np.shape(xdot))
    #robot_qdot= self.xdot2qdot(xdot, self.robot_state["q"]) #qdot
    # print("SHAPIN UP",np.shape(robot_qdot[0,0:6]))
        #= .001*(robotq[:7] - current_q[:7])