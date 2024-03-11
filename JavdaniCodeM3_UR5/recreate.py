#!/usr/bin/env python
import numpy as np
#Sets up the robot, environment, and goals to use shared autonomy code for grasping objects
from AssistanceHandler import *
from Primary_AssistPolicy import goal_from_object
from Goal import Goal
from Utils import *
#Choice of Model Env
from env3 import SimpleEnv
#Outer tools
from tf import *
from functools import partial
  #import adapy

#import prpy





env = SimpleEnv(visualize=False)
users = [1,2,3,4,5,6,7,8,9,10]
for j in users:
  for i in range(3):
      #goals, goal_objects = Initialize_Goals(env, robot, randomize_goal_init=False)
      goals, goal_objects = Initialize_Goals(env, randomize_goal_init=False)
      #print("HOT POTATO")
      ada_handler = AdaHandler(env, goals, goal_objects,user=j,demo= i+7) #goal objects is env objects, goals are GOAL object made from env objects
      #ada_handler.execute_policy(direct_teleop_only=False, fix_magnitude_user_command=True,w_comms=True)
      ada_handler.execute_policy_recreate(direct_teleop_only=False, fix_magnitude_user_command=True,w_comms=False)
      #ada_handler.execute_policy_simControlled(direct_teleop_only=False, fix_magnitude_user_command=False,w_comms=True)
      #ada_handler.execute_pol  -+icy_sim(direct_teleop_only=False, fix_magnitude_user_command=True,w_comms=True)
      #ada_handler.execute_direct_teleop(simulate_user=False)
  print("DONE DONE DONE")




# TODO: 
    #Change Scenarios to incentivize use of feedback interface
    