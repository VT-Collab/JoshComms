#Handles converting openrave items to generic assistance policy
from Goal import *
import BayesGoalPredictorV2 as GoalPredictor
#from ada_teleoperation.RobotState import Action

from Secondary_AssistPolicy import *
#from OpenraveUtils import *
import math
import numpy as np
import time

ADD_MORE_IK_SOLS = False

class PrimaryPolicy:

  def __init__(self, goals):
    self.goals = goals
    self.assist_policy = SecondaryPolicy(goals)
    self.goal_predictor = GoalPredictor.GoalPredictor(goals)
    self.inner_condifdence_step = .02
    self.alpha = 0
    

  def update(self, robot_state, user_action):
    
    goal_distribution = self.goal_predictor.get_distribution()
    self.assist_policy.update(robot_state, user_action,goal_distribution)
    self.user_action = user_action

    BaseQValues,values = self.assist_policy.get_values()
    self.goal_predictor.update_distribution(BaseQValues,values,user_action,robot_state)
    
    self.robot_state = robot_state
    

    


  #uses diffent goal distribution
  def get_blend_action(self, goal_distribution = np.array([]), **kwargs):
    if self.blend_confidence_function_prob_diff(goal_distribution, prob_diff_required=0.8):
      if self.alpha + self.inner_condifdence_step < 1:
        add_alpha = 0
        max_prob_goal_ind = np.argmax(goal_distribution)
        max_policy = self.assist_policy.goal_assist_policies[max_prob_goal_ind]
        max_QVpolicy = max_policy.target_assist_policies[0]
        #print(max_policy.target_assist_policies)
        max_assistConstant = max_QVpolicy.action_confirmation_constant
        if max_assistConstant < 1:
          add_alpha += (.25*(1-max_assistConstant))
        add_alpha += (.25*goal_distribution[max_prob_goal_ind])
        if np.linalg.norm(self.user_action) < .05:
          add_alpha += .25
        self.alpha += add_alpha*(self.inner_condifdence_step)
        #print("addition",add_alpha*(self.inner_condifdence_step))
        #print("Alpha,",self.alpha)

    else:
      self.alpha = .1
    

    assisted_qdot = self.assist_policy.get_assisted_action(goal_distribution, alpha = self.alpha,**kwargs)
    #print("YIPPEE")
    return assisted_qdot



  # #uses diffent goal distribution
  # def get_blend_action(self, goal_distribution = np.array([]), **kwargs):
    
  #   if goal_distribution.size == 0:
  #     goal_distribution = self.goal_predictor.get_distribution()
  #   max_prob_goal_ind = np.argmax(goal_distribution)


  #   if self.blend_confidence_function_prob_diff(goal_distribution): #confidence function requires cutoff range to be 40% more likely than the 2nd most likely probability
  #     goal_distribution_all_max = np.zeros(len(goal_distribution))
  #     goal_distribution_all_max[max_prob_goal_ind] = 1.0
  #     assisted_qdot = self.assist_policy.get_assisted_action(goal_distribution_all_max, **kwargs)
     
  #     return assisted_qdot
  #   else:
  #     #if we don't meet confidence function, use direct teleop
  #     return self.assist_policy.user_action
    


  def blend_confidence_function_prob_diff(self,goal_distribution, prob_diff_required=0.2):
    if len(goal_distribution) <= 1:
      print("FAILURE")
      return True

    goal_distribution_sorted = np.sort(goal_distribution)
   
    return goal_distribution_sorted[-1] - goal_distribution_sorted[-2] > prob_diff_required

  # manip_pos = robot_state.get_pos()
  # goal_poses = goal.target_poses
  # goal_pose_distances = [np.linalg.norm(manip_pos - pose[0:3,3]) for pose in goal_poses]
  # dist_to_nearest_goal = np.min(goal_pose_distances)
  # return dist_to_nearest_goal < distance_thresh

# def blend_confidence_function_euclidean_distance(robot_state, goals , distance_thresh=0.10):
#   manip_pos = (robot_state["x"])[0:2]
#   goal_pose_distances =[]
#   for goal in goals:
#     #goal_pos = goal.pos
#     goal_pose_distances = np.append(goal_pose_distances,[np.linalg.norm(manip_pos - goal.pos)])

#   dist_to_nearest_goal = np.min(goal_pose_distances)
#   return dist_to_nearest_goal < distance_thresh

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

