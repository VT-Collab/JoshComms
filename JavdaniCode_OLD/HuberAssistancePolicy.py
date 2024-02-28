import numpy as np
#import IPython
import AssistancePolicyOneTarget
from Utils import *


ACTION_DIMENSION = 7

class HuberAssistancePolicy(AssistancePolicyOneTarget.AssistancePolicyOneTarget):
  def __init__(self, goal):
    super(HuberAssistancePolicy, self).__init__(goal)
    self.set_constants(self.TRANSLATION_LINEAR_MULTIPLIER, self.TRANSLATION_DELTA_SWITCH, self.TRANSLATION_CONSTANT_ADD, self.ROTATION_LINEAR_MULTIPLIER,
                        self.ROTATION_DELTA_SWITCH, self.ROTATION_CONSTANT_ADD, self.ROTATION_MULTIPLIER)

    self.goal = goal
    self.goal_pos = self.goal.pos
    self.goal_quat = self.goal.quat
    

  def update(self, robot_state, user_action):
    super(HuberAssistancePolicy, self).update(robot_state, user_action)
    self.position_after_action,self.pose_after_action = joint2pose(self.robot_state_after_action)
    self.robot_position,self.robot_pose = joint2pose(self.robot_state["q"])

    
    self.dist_translation = np.linalg.norm(self.robot_position - self.goal_pos)
    self.dist_translation_aftertrans = np.linalg.norm(self.position_after_action - self.goal_pos)

    self.quat_curr = transmethods.quaternion_from_matrix(self.robot_pose)
    self.quat_after_trans = transmethods.quaternion_from_matrix(self.pose_after_action)

    #print("QUAT AND ROT",self.quat_curr)
    #print("Goal and bowl",self.goal_quat)
    self.dist_rotation = QuaternionDistance(self.quat_curr, self.goal_quat)
    
    self.dist_rotation_aftertrans = QuaternionDistance(self.quat_after_trans, self.goal_quat)
    
    #something about mode switch distance???

  def get_action(self):
    #q_rot = self.get_qderivative_rotation()
    #q_trans = self.get_qderivative_translation()
    current_x = self.robot_state["x"]
    goal_euler= np.array(transmethods.euler_from_quaternion(self.goal_quat))
    goal_x = np.append(self.goal_pos,goal_euler)
    xdot =  goal_x - current_x

    robot_qdot= xdot2qdot(xdot, self.robot_state) #qdot
    #print(np.shape(current_q[:7]),np.shape(robotq[:7]))
     #= .001*(robotq[:7] - current_q[:7])
    return robot_qdot

#   def get_qderivative_translation(self):
#     translation_diff = self.robot_position - self.goal_pos
#     dist_to_go = np.linalg.norm(translation_diff)
    
#     if dist_to_go > self.TRANSLATION_DELTA_SWITCH:
#       translation_derivative = self.TRANSLATION_LINEAR_COST_MULT_TOTAL*(translation_diff/dist_to_go)
#     else:
#       translation_derivative = self.TRANSLATION_CONSTANT_ADD*(translation_diff/dist_to_go)
#       translation_derivative += self.TRANSLATION_QUADRATIC_COST_MULTPLIER*translation_diff

#     #hacky part to make it not jumpy
#     dist_translation_limit = 2e-2
#     if (dist_to_go < dist_translation_limit):
#       translation_derivative *= dist_to_go/dist_translation_limit

#     return translation_derivative / self.ROBOT_TRANSLATION_COST_MULTIPLIER  #[1,3]
  
#   def get_qderivative_rotation(self):
#     quat_between = transmethods.quaternion_multiply(self.goal_quat, transmethods.quaternion_inverse(self.quat_after_trans))

#     rotation_derivative = quat_between[0:-1]
#     rotation_derivative /= np.linalg.norm(rotation_derivative)

#     if self.dist_rotation_aftertrans > self.ROTATION_DELTA_SWITCH:
#       rotation_derivative_magnitude = self.ROTATION_LINEAR_COST_MULT_TOTAL
#     else:
#       rotation_derivative_magnitude = self.ROTATION_CONSTANT_ADD
#       rotation_derivative_magnitude += self.ROTATION_QUADRATIC_COST_MULTPLIER*self.dist_rotation_aftertrans

#     rotation_derivative *= self.ROTATION_MULTIPLIER * rotation_derivative_magnitude / self.ROBOT_ROTATION_COST_MULTIPLIER

#     if (np.sum(self.goal_quat * self.quat_after_trans)) > 0:
#       rotation_derivative *= -1
#     #hacky part to make it not jumpy
#     dist_rotation_limit = np.pi/12.
#     #print 'dist rotation: ' + str(self.dist_rotation_aftertrans)
#     if (self.dist_rotation_aftertrans < dist_rotation_limit):
#       rotation_derivative *= self.dist_rotation_aftertrans/dist_rotation_limit
#     return rotation_derivative

#     #check to make sure direction is correct
# #    quat_thisdir = transition_quaternion(self.quat_curr, rotation_derivative, self.ACTION_APPLY_TIME)
# #    val_thisdir = self.get_value_rotation(dist_rotation=QuaternionDistance(quat_thisdir, self.goal_quat))
# #    quat_negdir = transition_quaternion(self.quat_curr, -1.*rotation_derivative, self.ACTION_APPLY_TIME)
# #    val_negdir = self.get_value_rotation(dist_rotation=QuaternionDistance(quat_negdir, self.goal_quat))
# #    if (val_thisdir < val_negdir):
# #      print 'ROT DIRECTION ERROR vals this dir: ' + str(val_thisdir) + '  neg dir: ' + str(val_negdir)
# #    else:
# #      print 'rotdir good'




#   def get_cost(self):
#     return self.get_cost_translation() + self.get_cost_rotation()

  def get_value(self):
    # a = self.get_value_translation()
    # print("Val-Trans: ",a)
    # b = self.get_value_rotation()
    # print("Val_Rot: ",b)
    # print("VAL TOTAL",a+b)
    # return a + b
    return (self.get_value_translation() + self.get_value_rotation())*self.V_Change_Constant

  def get_qvalue(self):
    # a = self.get_qvalue_translation()
    # print("QVal-Trans: ",a)
    # b = self.get_qvalue_rotation()
    # print("QVal_Rot: ",b)
    # print("QVAL TOTAL",a+b)
    # return a + b
    return (self.get_qvalue_translation() + self.get_qvalue_rotation())*self.Q_Change_Constant


    


#   #parts split into translation and rotation
  def get_value_translation(self, dist_translation=None):
    if dist_translation is None:
      dist_translation = self.dist_translation

    if dist_translation <= self.TRANSLATION_DELTA_SWITCH:
      return self.TRANSLATION_QUADRATIC_COST_MULTPLIER_HALF * dist_translation*dist_translation + self.TRANSLATION_CONSTANT_ADD*dist_translation;
    else:
      return self.TRANSLATION_LINEAR_COST_MULT_TOTAL * dist_translation - self.TRANSLATION_LINEAR_COST_SUBTRACT

  def get_cost_translation(self, dist_translation=None):
    if dist_translation is None:
      dist_translation = self.dist_translation

    if dist_translation > self.TRANSLATION_DELTA_SWITCH:
      return self.ACTION_APPLY_TIME * (self.TRANSLATION_LINEAR_COST_MULT_TOTAL)
    else:
      return self.ACTION_APPLY_TIME * (self.TRANSLATION_QUADRATIC_COST_MULTPLIER * dist_translation + self.TRANSLATION_CONSTANT_ADD)

  def get_qvalue_translation(self):
    return self.get_cost_translation() + self.get_value_translation(self.dist_translation_aftertrans)

  



  def get_value_rotation(self, dist_rotation=None):
    if dist_rotation is None:
      dist_rotation = self.dist_rotation

    if dist_rotation <= self.ROTATION_DELTA_SWITCH:
      return self.ROTATION_MULTIPLIER * (self.ROTATION_QUADRATIC_COST_MULTPLIER_HALF * dist_rotation*dist_rotation + self.ROTATION_CONSTANT_ADD*dist_rotation)
    else:
      return self.ROTATION_MULTIPLIER*(self.ROTATION_LINEAR_COST_MULT_TOTAL * dist_rotation - self.ROTATION_LINEAR_COST_SUBTRACT)

  def get_cost_rotation(self, dist_rotation=None):
    if dist_rotation is None:
      dist_rotation = self.dist_rotation

    if dist_rotation > self.ROTATION_DELTA_SWITCH:
      return self.ACTION_APPLY_TIME * self.ROTATION_MULTIPLIER * self.ROTATION_LINEAR_COST_MULT_TOTAL
    else:
      return self.ACTION_APPLY_TIME * self.ROTATION_MULTIPLIER * (self.ROTATION_QUADRATIC_COST_MULTPLIER * dist_rotation + self.ROTATION_CONSTANT_ADD)

  def get_qvalue_rotation(self):
    return self.get_cost_rotation() + self.get_value_rotation(self.dist_rotation_aftertrans)

  




  #HUBER CONSTANTS
  #Values used when assistance always on
  TRANSLATION_LINEAR_MULTIPLIER = 2.25
  TRANSLATION_DELTA_SWITCH = 0.07
  TRANSLATION_CONSTANT_ADD = 0.2

  ROTATION_LINEAR_MULTIPLIER = 0.20
  #ROTATION_DELTA_SWITCH = np.pi/7.
  ROTATION_DELTA_SWITCH = np.pi/32. #.0981
  ROTATION_CONSTANT_ADD = 0.01
  ROTATION_MULTIPLIER = 0.07

  #ROBOT_TRANSLATION_COST_MULTIPLIER = 14.5
  #ROBOT_ROTATION_COST_MULTIPLIER = 0.10

  ROBOT_TRANSLATION_COST_MULTIPLIER = 50.0
  ROBOT_ROTATION_COST_MULTIPLIER = 0.05



  #HUBER CACHED CONSTANTS that will be calculated soon
  TRANSLATION_LINEAR_COST_MULT_TOTAL = 0.0
  TRANSLATION_QUADRATIC_COST_MULTPLIER = 0.0
  TRANSLATION_QUADRATIC_COST_MULTPLIER_HALF = 0.0
  TRANSLATION_LINEAR_COST_SUBTRACT = 0.0

  ROTATION_LINEAR_COST_MULT_TOTAL = 0.0
  ROTATION_QUADRATIC_COST_MULTPLIER = 0.0
  ROTATION_QUADRATIC_COST_MULTPLIER_HALF = 0.0
  ROTATION_LINEAR_COST_SUBTRACT = 0.0



  def set_constants(self, huber_translation_linear_multiplier, huber_translation_delta_switch, huber_translation_constant_add, huber_rotation_linear_multiplier, huber_rotation_delta_switch, 
                    huber_rotation_constant_add, huber_rotation_multiplier, Change_Constant = 1,robot_translation_cost_multiplier=None, robot_rotation_cost_multiplier=None):
    self.TRANSLATION_LINEAR_MULTIPLIER = huber_translation_linear_multiplier
    self.TRANSLATION_DELTA_SWITCH = huber_translation_delta_switch
    self.TRANSLATION_CONSTANT_ADD = huber_translation_constant_add

    self.ROTATION_LINEAR_MULTIPLIER = huber_rotation_linear_multiplier
    self.ROTATION_DELTA_SWITCH = huber_rotation_delta_switch
    self.ROTATION_CONSTANT_ADD = huber_rotation_constant_add
    self.ROTATION_MULTIPLIER = huber_rotation_multiplier
    self.V_Change_Constant = Change_Constant# Josh Addition, Slows Change moderation. Rerouted into Goal Predictor for goal bypass in redirection cases
    self.Q_Change_Constant = Change_Constant#Josh Addition, 
    if robot_translation_cost_multiplier:
      self.ROBOT_TRANSLATION_COST_MULTIPLIER = robot_translation_cost_multiplier
    if robot_rotation_cost_multiplier:
      self.ROBOT_ROTATION_COST_MULTIPLIER = robot_rotation_cost_multiplier
    
    self.calculate_cached_constants()

  #other constants that are cached for faster computation
  def calculate_cached_constants(self):
    self.TRANSLATION_LINEAR_COST_MULT_TOTAL = self.TRANSLATION_LINEAR_MULTIPLIER + self.TRANSLATION_CONSTANT_ADD #2.45
    self.TRANSLATION_QUADRATIC_COST_MULTPLIER = self.TRANSLATION_LINEAR_MULTIPLIER/self.TRANSLATION_DELTA_SWITCH #2.25 / .07 = 32.1423
    self.TRANSLATION_QUADRATIC_COST_MULTPLIER_HALF = 0.5 * self.TRANSLATION_QUADRATIC_COST_MULTPLIER
    self.TRANSLATION_LINEAR_COST_SUBTRACT = self.TRANSLATION_LINEAR_MULTIPLIER * self.TRANSLATION_DELTA_SWITCH * 0.5 #.5*2.25*.07 = 0.07875

    self.ROTATION_LINEAR_COST_MULT_TOTAL = self.ROTATION_LINEAR_MULTIPLIER + self.ROTATION_CONSTANT_ADD #0.2 + 0.01 = 0.21
    self.ROTATION_QUADRATIC_COST_MULTPLIER = self.ROTATION_LINEAR_MULTIPLIER/self.ROTATION_DELTA_SWITCH #.2 / .0981 = 2.0387
    self.ROTATION_QUADRATIC_COST_MULTPLIER_HALF = 0.5 * self.ROTATION_QUADRATIC_COST_MULTPLIER #1.0194
    self.ROTATION_LINEAR_COST_SUBTRACT = self.ROTATION_LINEAR_MULTIPLIER * self.ROTATION_DELTA_SWITCH * 0.5 #0.2 * .0981 *.5 = 0.00981


# def UserInputToRobotAction(user_input):
#   return np.append(user_input, np.zeros(3))


# def transition_quaternion(quat, angular_vel, action_apply_time):
#   norm_vel = np.linalg.norm(angular_vel)
#   return transmethods.quaternion_multiply( transmethods.quaternion_about_axis(action_apply_time*norm_vel, angular_vel/norm_vel) , quat)