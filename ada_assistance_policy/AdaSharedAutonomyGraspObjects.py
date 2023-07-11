#!/usr/bin/env python

#Sets up the robot, environment, and goals to use shared autonomy code for grasping objects

from AdaHandler import *
#from ada_teleoperation.AdaTeleopHandler import possible_teleop_interface_names
from AdaAssistancePolicy import goal_from_object
#import rospy
#import rospkg
#import IPython

import numpy as np
from Goal import Goal
from functools import partial



#import adapy
from env import SimpleEnv
#import prpy


VIEWER_DEFAULT = 'InteractiveMarker'

#def Initialize_Adapy(args, env_path='/environments/tablewithobjects_assisttest.env.xml'):
def Initialize_Env():
    env = SimpleEnv(visualize=True)
    #Init_Robot(robot)
    
    return env

def Initialize_Goals(env,  randomize_goal_init=False):
  while True:
    goals, goal_objects = Init_Goals(env, randomize_goal_init)

    #check if we have sufficient poses for each goal, to see if initialization succeeded
    for goal in goals:
      if len(goal.target_poses) < 25:
        continue
    break

  return goals, goal_objects


def Init_Goals(env, robot, randomize_goal_init=False):
    #TODO: Make skill goal pairs for each of tasks
    goal_objects = []

    
    goal_objects.append(env.block1)

    
    if randomize_goal_init:
      env.block_position += np.random.rand(3)*0.10 - 0.05
      env.reset_box()


    goals = Set_Goals_From_Objects(env,goal_objects)

    #for obj in goal_objects:
    #  obj.Enable(True)

    return goals, goal_objects


def Set_Goals_From_Objects(env,goal_objects):

#    else:
  goals = []
  for obj in goal_objects:
    goals.append(goal_from_object(env,obj))


  return goals

def goal_from_object(env,object):
  #pose = object.GetTransform()
  #robot = manip.GetRobot()

  num_poses_desired = 30
  max_num_poses_sample = 500

  target_poses = []
  target_iks = []
  num_sampled = 0
  manip = env.panda
  obj_pos = object.get_position()
  obj_quat = object.get_orientation()
  #print(pose)
  ik_sol = manip._inverse_kinematics(obj_pos, [0]*4)
  target_poses.append(obj_quat)
  target_iks.append(ik_sol)
  print(ik_sol)
  return Goal(obj_quat,obj_pos , target_poses = target_poses, target_iks = target_iks)



def Finish_Trial_Func(robot):
  if robot.simulated:
    num_hand_dofs = len(robot.arm.hand.GetDOFValues())
    robot.arm.hand.SetDOFValues(np.ones(num_hand_dofs)*0.8)
  else:
    robot.arm.hand.CloseHand()

def Reset_Robot(robot):
  if robot.simulated:
    num_hand_dofs = len(robot.arm.hand.GetDOFValues())
    inds, pos = robot.configurations.get_configuration('home')
    with robot.GetEnv():
      robot.SetDOFValues(pos, inds)
      robot.arm.hand.SetDOFValues(np.ones(num_hand_dofs)*0.1)
  else:
    robot.arm.hand.OpenHand()
    robot.arm.PlanToNamedConfiguration('home', execute=True)



if __name__ == "__main__":
  parser = argparse.ArgumentParser('Ada Assistance Policy')
  # parser.add_argument('-s', '--sim', action='store_true', default=SIMULATE_DEFAULT,
  #                     help='simulation mode')
  # parser.add_argument('-v', '--viewer', nargs='?', const=True, default=VIEWER_DEFAULT,
  #                     help='attach a viewer of the specified type')
  #parser.add_argument('--env-xml', type=str,
                      #help='environment XML file; defaults to an empty environment')
  parser.add_argument('--debug', action='store_true',
                      help='enable debug logging')
  #parser.add_argument('-input', '--input-interface-name', help='name of the input interface. Possible choices: ' + str(possible_teleop_interface_names), type=str)
  parser.add_argument('-joy_dofs', '--num-input-dofs', help='number of dofs of input, either 2 or 3', type=int, default=2)
  args = parser.parse_args()

  #.init_node('ada_assistance_policy', anonymous = True)
  
  env = Initialize_Env()
    #env,robot = Initialize_Adapy(args, env_path=env_path)
  time.sleep(30)
  #finish_trial_func_withrobot = partial(Finish_Trial_Func, robot=robot)
  #
  for i in range(1):
    #goals, goal_objects = Initialize_Goals(env, robot, randomize_goal_init=False)
    goals, goal_objects = Initialize_Goals(env, randomize_goal_init=False)
    ada_handler = AdaHandler(env, goals, goal_objects) #goal objects is env objects, goals are GOAL object made from env objects
    ada_handler.execute_policy(simulate_user=True, direct_teleop_only=False, fix_magnitude_user_command=False)
  #ada_handler.execute_direct_teleop(simulate_user=False)



