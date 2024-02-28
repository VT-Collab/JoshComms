#WE DON'T EXPECT TO USE THIS CLASS, BUT RATHER ONE THAT INHERITS FROM IT
#Generic assistance policy for one target
import numpy as np
#import IPython
#import AssistancePolicyOneTarget as TargetPolicy
#import HuberAssistancePolicy as TargetPolicy
import Basic_QV_V1 as TargetPolicy

TargetPolicyClass = TargetPolicy.HuberAssistancePolicy

class AssistancePolicyOneGoal:
  def __init__(self, goal):
    self.goal = goal
    #self.panda = panda 

    self.target_assist_policies = []
    for pose in self.goal.target_poses:
      #a = self.goal.pos
      
      self.target_assist_policies.append(TargetPolicyClass(goal))
      
    self.min_val_ind = 0

  def update(self, robot_state, user_action,max_goal_pos=None,goal_distrib=[],Robot_action = np.zeros(6)):
    self.last_robot_state = robot_state
    self.last_user_action = user_action

    for target_policy in self.target_assist_policies:
      target_policy.update(robot_state, user_action,max_goal_pos,goal_distrib,Robot_action)

    values = [targ_policy.get_value() for targ_policy in self.target_assist_policies]
    self.min_val_ind = np.argmin(values)

  def get_value(self):
    return self.target_assist_policies[self.min_val_ind].get_value()

  def get_qvalue(self):
    return self.target_assist_policies[self.min_val_ind].get_qvalue()
  
  def get_BaseQValue(self):
    return self.target_assist_policies[self.min_val_ind].base_Q()

  def get_action(self):
    values = [targ_policy.get_value() for targ_policy in self.target_assist_policies] #get action value dependent on linear and rotational distance to robot
    min_val_ind = np.argmin(values) #get min distance move choice
    return self.target_assist_policies[min_val_ind].get_action() #pick smallest action

# def get_value(self):
    #return self.get_value_translation() + self.get_value_rotation()
    #(Dist of EE to  Goal) == d, Quarterion distance between current and final pose = rd
    #Get value trans = USE IF FURTHER AWAY: (.2*d) -  0.07875 or #USE IF CLOSE : (11.459*(d^2)) + (.2*d)
    #Rot version: (.07* (0.21*rd - 0.00981) )  or .07*1.0194*(rd^2) + (.01*rd) 
    #if close: (11.459*(d^2)) + (.2*d) + .07*1.0194*(rd^2) + (.01*rd) 
    #else: ((.2*d) -  0.07875) + (.07* (0.21*rd - 0.00981) )

  # #parts split into translation and rotation
  # def get_value_translation(self, dist_translation=None):
  #   if dist_translation is None:
  #     dist_translation = self.dist_translation #self dist is the distance between EE and goal

  #   if dist_translation <= self.TRANSLATION_DELTA_SWITCH: #in built constant, base is TRANSLATION_DELTA_SWITCH = 0.07, TRANSLATION_CONSTANT_ADD = 0.2, Quadratic Cost = (2.25/(np.pi/32.))/2. = 11.459
  #     return self.TRANSLATION_QUADRATIC_COST_MULTPLIER_HALF * dist_translation*dist_translation + self.TRANSLATION_CONSTANT_ADD*dist_translation;
                #USE IF CLOSE : (11.459*(d^2)) + (.2*d)
  #   else:
  #     return self.TRANSLATION_LINEAR_COST_MULT_TOTAL * dist_translation - self.TRANSLATION_LINEAR_COST_SUBTRACT
          # USE IF FURTHER AWAY: (.2*d) -  0.07875

# def get_value_rotation(self, dist_rotation=None):
#     if dist_rotation is None:
#       dist_rotation = self.dist_rotation #difference in quarterion angles

#     if dist_rotation <= self.ROTATION_DELTA_SWITCH: # self.ROTATION_DELTA_SWITCH = .0981
#       return self.ROTATION_MULTIPLIER * (self.ROTATION_QUADRATIC_COST_MULTPLIER_HALF * dist_rotation*dist_rotation + self.ROTATION_CONSTANT_ADD*dist_rotation) 
        # self.ROTATION_MULTIPLIER = 0.07, ROTATION_QUADRATIC_COST_MULTPLIER_HALF = 11.4679, ROTATION_CONSTANT_ADD = 0.01
        # .07*1.0194*(rd^2) + (.01*rd)
#     else:
#       return self.ROTATION_MULTIPLIER*(self.ROTATION_LINEAR_COST_MULT_TOTAL * dist_rotation - self.ROTATION_LINEAR_COST_SUBTRACT) 
        #(.07* (0.21*rd - 0.00981) ) 
    

  def get_min_value_pose(self):
    return self.goal.target_poses[self.min_val_ind]
