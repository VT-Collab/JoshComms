#!/usr/bin/env python
import numpy as np
#Sets up the robot, environment, and goals to use shared autonomy code for grasping objects
from AssistanceHandler import *
from Primary_AssistPolicy import goal_from_object
from Goal import Goal
from Utils import *
#Choice of Model Env
from env2 import SimpleEnv
#Outer tools
from tf import *
from functools import partial
#import adap
#import prpy



if __name__ == "__main__":
  parser = argparse.ArgumentParser('Ada Assistance Policy')
  parser.add_argument('--debug', action='store_true',
                      help='enable debug logging')
  #parser.add_argument('-input', '--input-interface-name', help='name of the input interface. Possible choices: ' + str(possible_teleop_interface_names), type=str)
  parser.add_argument('-joy_dofs', '--num-input-dofs', help='number of dofs of input, either 2 or 3', type=int, default=2)
  args = parser.parse_args()

  env = SimpleEnv(visualize=True)

  for i in range(1):
    #goals, goal_objects = Initialize_Goals(env, robot, randomize_goal_init=False)
    goals, goal_objects = Initialize_Goals(env, randomize_goal_init=False)
    #print("HOT POTATO")
    ada_handler = AdaHandler(env, goals, goal_objects,user=200,demo=1) #goal objects is env objects, goals are GOAL object made from env objects
    #ada_handler.execute_policy(direct_teleop_only=False, fix_magnitude_user_command=False,w_comms=True)
    #ada_handler.execute_policy_simControlled(direct_teleop_only=False, fix_magnitude_user_command=False,w_comms=False,algo_enabled = True)
    ada_handler.execute_policy_sim(direct_teleop_only=False, fix_magnitude_user_command=False,w_comms=False,algo_enabled = True)
  #ada_handler.execute_direct_teleop(simulate_user=False)





# TODO: 
    #Change Scenarios to incentivize use of feedback interface
    