#!/usr/bin/env python

#Sets up the robot, environment, and goals to use shared autonomy code for grasping objects

from AdaHandler import *
from ada_teleoperation.AdaTeleopHandler import possible_teleop_interface_names
from AdaAssistancePolicy import goal_from_object
import rospy
import rospkg
import IPython

import numpy as np

from functools import partial


import openravepy
#import adapy
from env import SimpleEnv
import prpy


VIEWER_DEFAULT = 'InteractiveMarker'

#def Initialize_Adapy(args, env_path='/environments/tablewithobjects_assisttest.env.xml'):
def Initialize_Adapy():
    """ Initializes robot and environment through adapy, using the specified environment path

    @param env_path path to OpenRAVE environment
    @return environment, robot
    """
    #env_path = '/environments/tablewithobjects_assisttest.env.xml'
    # adapy_args = {'sim':args.sim,
    #               'attach_viewer':args.viewer,
    #               'env_path':env_path
    #               }
    # openravepy.RaveInitialize(True, level=openravepy.DebugLevel.Debug)
    #openravepy.misc.InitOpenRAVELogging();
    # def Init_Robot(robot):
#     robot.SetActiveDOFs(range(6))
#     #self.robot.arm.hand.OpenHand()
#     #if (self.sim):
#     robot_pose = np.array([[1, 0, 0, 0.409],[0, 1, 0, 0.338],[0, 0, 1, 0.795],[0, 0, 0, 1]])
#     robot.SetTransform(robot_pose)
#     if (robot.simulated):
#       #servo simulator params
#       robot.arm.servo_simulator.period=1./200.
    #env, robot = adapy.initialize(**adapy_args)
    env = SimpleEnv(visualize=False)
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
    goal_objects = []
    goal_objects.append(env.block)
    #env.block_position = [0.5, 0.4, 0.2]
    #env.block_quaternion = [0, 0, 0, 1]
    #goal_objects.append(env.GetKinBody('fuze_bottle'))
    

    if randomize_goal_init:
      env.block_position += np.random.rand(3)*0.10 - 0.05
      env.reset_box()
    

   

    # if randomize_goal_init:
    #   obj_pose[0:3,3] += np.random.rand(3)*0.10 - 0.05
    # goal_objects[1].SetTransform(obj_pose)
    # obj_pose= robot.GetTransform()

    # #disable collisions for IK
    # for obj in goal_objects:
    #   obj.Enable(False)

    goals = Set_Goals_From_Objects(env,goal_objects)

    #for obj in goal_objects:
    #  obj.Enable(True)

    return goals, goal_objects


def Set_Goals_From_Objects(env,goal_objects):
  #construct filename where data might be
#    path_to_pkg = rospkg.RosPack().get_path('ada_assistance_policy')
#    filename = path_to_pkg + "/" + cached_data_dir + "/"
#    for obj in goal_objects:
#      filename += str(obj.GetName())
#      str_pos = str(obj.GetTransform()[0:3, -1])[2:-2]
#      str_pos = str_pos.replace(" ","")
#      str_pos = str_pos.replace(".","")
#      filename += str_pos
#    filename += '.pckl'
#
#    #see if cached, if not load and save
#    if os.path.isfile(filename) and not RESAVE_GRASP_POSES:
#      with open(filename, 'r') as file:
#        items = pickle.load(file)
#        goals = items['goals']
#    else:
  goals = []
  for obj in goal_objects:
    goals.append(goal_from_object(env,obj))
#      with open(filename, 'w') as file:
#        items = {}
#        items['goals'] = goals
#        pickle.dump(items, file)

  return goals

def goal_from_object(env,object):
  #pose = object.GetTransform()
  #robot = manip.GetRobot()
  

  #generate TSRs for object
  

  #turn TSR into poses
  num_poses_desired = 30
  max_num_poses_sample = 500

  target_poses = []
  target_iks = []
  num_sampled = 0

  obj_pos = object.get_position()
  pose = object.get_orientation()
  while len(target_poses) < num_poses_desired and num_sampled < max_num_poses_sample:
    for pose in target_poses_tocheck:
      #check if solution exists
#      ik_sols = manip.FindIKSolutions(pose, openravepy.IkFilterOptions.CheckEnvCollisions)
#      if len(ik_sols) > 0:

      
      #sample some random joint vals
      
#      lower, upper = robot.GetDOFLimits()
#      dof_vals_before = robot.GetActiveDOFValues()
#      dof_vals = [ np.random.uniform(lower[i], upper[i]) for i in range(6)]
#      robot.SetActiveDOFValues(dof_vals)
#      pose = manip.GetEndEffectorTransform()
#      robot.SetActiveDOFValues(dof_vals_before)

      ik_sol = manip.FindIKSolution(pose, openravepy.IkFilterOptions.CheckEnvCollisions)
      if ik_sol is not None:
        if ADD_MORE_IK_SOLS:
          #get bigger list of ik solutions
          ik_sols = manip.FindIKSolutions(pose, openravepy.IkFilterOptions.CheckEnvCollisions)
          if ik_sols is None:
            ik_sols = list()
          else:
            ik_sols = list(ik_sols)
          #add the solution we found before
          ik_sols.append(ik_sol)
        else:
          #if we don't want to add more, just use the one we found
          ik_sols = [ik_sol]
        #check env col
        target_poses.append(pose)
        target_iks.append(ik_sols)
#        with robot:
#          manip.SetDOFValues(ik_sol)
#          if not env.CheckCollision(robot):
#            target_poses.append(pose)
        if len(target_poses) >= num_poses_desired:
          break

  return Goal(pose, target_poses, target_iks)



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
  parser.add_argument('-input', '--input-interface-name', help='name of the input interface. Possible choices: ' + str(possible_teleop_interface_names), type=str)
  parser.add_argument('-joy_dofs', '--num-input-dofs', help='number of dofs of input, either 2 or 3', type=int, default=2)
  args = parser.parse_args()

  rospy.init_node('ada_assistance_policy', anonymous = True)


  ##find environment path
  #path_to_pkg = rospkg.RosPack().get_path('ada_assistance_policy')
  # env_path = os.path.join(path_to_pkg, 'data', 'environments', 'tablewithobjects_assisttest.env.xml')

  
  env = Initialize_Adapy()
    #env,robot = Initialize_Adapy(args, env_path=env_path)

  #finish_trial_func_withrobot = partial(Finish_Trial_Func, robot=robot)
  #
  for i in range(1):
    #goals, goal_objects = Initialize_Goals(env, robot, randomize_goal_init=False)
    goals, goal_objects = Initialize_Goals(env, randomize_goal_init=False)
    ada_handler = AdaHandler(env, robot, goals, goal_objects, args.input_interface_name, args.num_input_dofs, use_finger_mode=False)
    ada_handler.execute_policy(simulate_user=False, direct_teleop_only=False, fix_magnitude_user_command=False, finish_trial_func=finish_trial_func_withrobot)
  #ada_handler.execute_direct_teleop(simulate_user=False)

  IPython.embed()

