#!/usr/bin/env python

#Sets up the robot, environment, and goals to use shared autonomy code for grasping objects

from AssistanceHandler import *
from Primary_AssistPolicy import goal_from_object

import numpy as np
from Goal import Goal
from Utils import *
from env3 import SimpleEnv

from tf import *
from functools import partial
#import adapy

#import prpy



if __name__ == "__main__":
  parser = argparse.ArgumentParser('Ada Assistance Policy')
  parser.add_argument('--debug', action='store_true',
                      help='enable debug logging')
  #parser.add_argument('-input', '--input-interface-name', help='name of the input interface. Possible choices: ' + str(possible_teleop_interface_names), type=str)
  parser.add_argument('-joy_dofs', '--num-input-dofs', help='number of dofs of input, either 2 or 3', type=int, default=2)
  args = parser.parse_args()

  env = Initialize_Env(visualize=False)

  for i in range(1):
    #goals, goal_objects = Initialize_Goals(env, robot, randomize_goal_init=False)
    goals, goal_objects = Initialize_Goals(env, randomize_goal_init=False)
    #print("HOT POTATO")
    ada_handler = AdaHandler(env, goals, goal_objects) #goal objects is env objects, goals are GOAL object made from env objects
    ada_handler.execute_policy(direct_teleop_only=False, fix_magnitude_user_command=False,w_comms=False,confVibe= True)
    #ada_handler.execute_policy_sim(direct_teleop_only=True, fix_magnitude_user_command=True,w_comms=False)
  #ada_handler.execute_direct_teleop(simulate_user=False)



