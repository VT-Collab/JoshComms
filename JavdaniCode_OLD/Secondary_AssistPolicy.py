#Generic assistance policy for one goal
import numpy as np
#import time
#import IPython
import AssistancePolicyOneGoal as GoalPolicy

class AssistancePolicy:

  def __init__(self, goals):
    self.goals = goals
    #self.panda = panda

    self.goal_assist_policies = []
    for goal in goals:
      self.goal_assist_policies.append(GoalPolicy.AssistancePolicyOneGoal(goal))

    #self.user_input_mapper = UserInputMapper()


  def update(self, robot_state, user_action):
    self.robot_state = robot_state
    #user action corresponds to the effect of direct teleoperation on the robot
    #self.user_action = self.user_input_mapper.input_to_action(user_input, robot_state)
    self.user_action = user_action

    for goal_policy in self.goal_assist_policies:
      goal_policy.update(robot_state, self.user_action)

  def get_values(self):
    values = np.ndarray(len(self.goal_assist_policies))
    qvalues = np.ndarray(len(self.goal_assist_policies))
    for ind,goal_policy in enumerate(self.goal_assist_policies):
      #print(goal_policy.goal.name,"_______________________________________________")
      values[ind] = goal_policy.get_value()
      qvalues[ind] = goal_policy.get_qvalue()

    return values,qvalues


  def get_probs_last_user_action(self):
    values,qvalues = self.get_values()
   
    return np.exp(-(qvalues-values))




  def get_assisted_action(self, goal_distribution, fix_magnitude_user_command=False,alpha=.4):
    assert goal_distribution.size == len(self.goal_assist_policies)

    action_dimension = 6
    #TODO how do we handle mode switch vs. not?
    total_action_twist = np.zeros(action_dimension)
    for goal_policy,goal_prob in zip(self.goal_assist_policies, goal_distribution):
      total_action_twist += goal_prob * goal_policy.get_action() #goal policy gets smallest action for the relevant goal given huber parameters, multiplies by probability level for 

    total_action_twist /= np.sum(goal_distribution)
    #user action is 1x7 joint space


    
    if np.linalg.norm(self.user_action)>.0005:
      UserAdjusted = self.user_action
      
    else:
      UserAdjusted = .5*total_action_twist + self.user_action
      
    to_ret_twist = 4*(total_action_twist*(alpha) + UserAdjusted*(1-alpha))

    if fix_magnitude_user_command:
      to_ret_twist*= np.linalg.norm(self.user_action)/np.linalg.norm(to_ret_twist)


    return to_ret_twist
    
      
 