import numpy as np
import tf as transmethods
import copy
from Utils import *

class AssistancePolicyOneTarget(object):
  ACTION_APPLY_TIME = .5

  def __init__(self, goal):
    self.goal_pose = goal.pose
    self.goal_quat = goal.pose
    self.goal_pos = goal.pos

  def update(self, robot_state, user_action):
    self.robot_state = copy.deepcopy(robot_state)
    self.user_action = copy.deepcopy(user_action)
    #print("USER",user_action)
    #self.robot_state_after_action = self.state_after_user_action(robot_state, user_action)
    self.rob_pos = self.robot_state["x"]
    #self.rob_pos = self.ee_trans[0:3,3]

  def get_action(self):
    
    pos_diff = 5.*(self.goal_pos - self.rob_pos)

    pos_diff_norm = np.linalg.norm(pos_diff)

    clip_norm_val = 0.02
    if (pos_diff_norm > clip_norm_val):
      pos_diff /= pos_diff_norm/clip_norm_val

    return pos_diff



  
  def state_after_user_action(self,robot_state,qdot, limit=1.0):
      
      
      qdot = np.asarray(qdot)
      #print(qdot)
      # scale = np.linalg.norm(qdot)
      # if scale > limit:
      #     qdot = np.asarray([qdot[i] * limit/scale for i in range(len(qdot))])
      current_q = robot_state["q"]
      qafter = current_q + (qdot*self.ACTION_APPLY_TIME)
      
      return qafter[0]


#def UserInputToRobotAction(user_input):
#  return user_input
