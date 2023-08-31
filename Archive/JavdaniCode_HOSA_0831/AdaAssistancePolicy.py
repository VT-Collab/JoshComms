#Handles converting openrave items to generic assistance policy
from Goal import *
import GoalPredictor as GoalPredictor
#from ada_teleoperation.RobotState import Action

from AssistancePolicy import *
#from OpenraveUtils import *
import math
import numpy as np
import time

ADD_MORE_IK_SOLS = False

class AdaAssistancePolicy:

  def __init__(self, goals):
    self.goals = goals
    self.assist_policy = AssistancePolicy(goals)
    self.goal_predictor = GoalPredictor.GoalPredictor(goals)
    

  def update(self, robot_state, user_action,panda):
    #print(self.goals[0].grasp)
    self.assist_policy.update(robot_state, user_action,panda)
    
    values,q_values = self.assist_policy.get_values()
    
    self.goal_predictor.update_distribution(values, q_values)
    self.robot_state = robot_state

  # def goal_update(self, robot_state, user_action,panda):
  #   self.goals = goals
  #   self.assist_policy = AssistancePolicy(goals)
  #   self.goal_predictor = GoalPredictor.GoalPredictor(goals)

  def get_action(self, goal_distribution = np.array([]), **kwargs):
    if goal_distribution.size == 0:
      goal_distribution = self.goal_predictor.get_distribution()
    #twist is qdot brought from a a mix of input and goal prediction, gets human action through internal update run
    assisted_qdot = self.assist_policy.get_assisted_action(goal_distribution, **kwargs)
    #assisted_action = twist=self.assist_policy.get_assisted_action(goal_distribution, **kwargs), switch_mode_to=self.assist_policy.user_action.switch_mode_to)
    #generates twist for assist policy representing the angular and linear velocity of each joint along the 7 points
    #switch modes dependent on cofidence level
    return assisted_qdot

  #uses diffent goal distribution
  def get_blend_action(self, goal_distribution = np.array([]), **kwargs):
    if goal_distribution.size == 0:
      goal_distribution = self.goal_predictor.get_distribution()

    max_prob_goal_ind = np.argmax(goal_distribution)


    #check if we meet the confidence criteria which dictates whether or not assistance is provided
    #use the one from ancas paper - euclidean distance and some threshhold
    #if blend_confidence_function_euclidean_distance(self.robot_state, self.goals[max_prob_goal_ind]):

    if blend_confidence_function_prob_diff(goal_distribution): #confidence function requires cutoff range to be 40% more likely than the 2nd most likely probability
      goal_distribution_all_max = np.zeros(len(goal_distribution))
      goal_distribution_all_max[max_prob_goal_ind] = 1.0
      #assisted_action = Action(twist=self.assist_policy.get_assisted_action(goal_distribution_all_max, **kwargs), switch_mode_to=self.assist_policy.user_action.switch_mode_to)
      assisted_qdot = self.assist_policy.get_assisted_action(goal_distribution, **kwargs)
      return assisted_qdot
    else:
      #if we don't meet confidence function, use direct teleop
      return self.assist_policy.user_action

def blend_confidence_function_prob_diff(goal_distribution, prob_diff_required=0.4):
  if len(goal_distribution) <= 1:
    return True

  goal_distribution_sorted = np.sort(goal_distribution)
  return goal_distribution_sorted[-1] - goal_distribution_sorted[-2] > prob_diff_required

  manip_pos = robot_state.get_pos()
  goal_poses = goal.target_poses
  goal_pose_distances = [np.linalg.norm(manip_pos - pose[0:3,3]) for pose in goal_poses]
  dist_to_nearest_goal = np.min(goal_pose_distances)
  return dist_to_nearest_goal < distance_thresh

def blend_confidence_function_euclidean_distance(robot_state, goal, distance_thresh=0.10):
  manip_pos = robot_state.get_pos()
  goal_poses = goal.target_poses
  goal_pose_distances = [np.linalg.norm(manip_pos - pose[0:3,3]) for pose in goal_poses]
  dist_to_nearest_goal = np.min(goal_pose_distances)
  return dist_to_nearest_goal < distance_thresh

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
  pose = object.get_orientation()
  ik_sol = manip._inverse_kinematics(obj_pos, dquaternion=[0]*4)
  target_poses.append(pose)
  target_iks.append(ik_sol)
  return Goal(pose, target_poses, target_iks)

