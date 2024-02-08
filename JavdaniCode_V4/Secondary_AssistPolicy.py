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
      #print(values[ind]-qvalues[ind])
    #time.sleep(20)

    return values,qvalues


  def get_probs_last_user_action(self):
    values,qvalues = self.get_values()
    #print np.exp(-(qvalues-values))
    return np.exp(-(qvalues-values))




  def get_assisted_action(self, goal_distribution, fix_magnitude_user_command=False):
    assert goal_distribution.size == len(self.goal_assist_policies)

    action_dimension = GoalPolicy.TargetPolicy.ACTION_DIMENSION
    #TODO how do we handle mode switch vs. not?
    total_action_twist = np.zeros(action_dimension)
    for goal_policy,goal_prob in zip(self.goal_assist_policies, goal_distribution):
      total_action_twist += goal_prob * goal_policy.get_action() #goal policy gets smallest action for the relevant goal given huber parameters, multiplies by probability level for magniutde

    total_action_twist /= np.sum(goal_distribution)
    #user action is 1x7 joint space
    #print("ROBOT",np.linalg.norm(total_action_twist))
    #print("USER",np.linalg.norm(self.user_action))
    
    #ratio = np.linalg.norm(self.user_action)/np.linalg.norm(total_action_twist)
    #print("#Ratio",ratio)
    #UserAdjusted = [item * 1 for item in self.user_action]
    UserAdjusted = self.user_action

    # dist2goal = np.linalg.norm((self.robot_state["x"])[0:3] - ((self.goals[np.argmax(goal_distribution)]).pos)[0:3])
    # #print("DIST TO GOAL",(dist2goal))
    # sorted = np.sort(goal_distribution)
    # diff = abs(sorted[-1] - sorted[-2])
    # #print("Diff Confidence",diff)
    # blend_mult = (diff/.98) + ((.52)/((dist2goal)+.52)) #.52 is starting distance from every goal

    if np.linalg.norm(self.user_action) > .01:
      dist2goal = np.linalg.norm((self.robot_state["x"])[0:3] - ((self.goals[np.argmax(goal_distribution)]).pos)[0:3])
      ratio = np.linalg.norm(self.user_action)/np.linalg.norm(total_action_twist)
      RobotAdjusted = [item * ratio*((.52)/((dist2goal**.5)+.52)) for item in total_action_twist]
      #print("ROBOT ADjusted",np.linalg.norm(RobotAdjusted))
      #print("#Ratio",ratio)
    else: 
      RobotAdjusted = total_action_twist*.75 #*blend_mult
    #print("ROBOT",np.linalg.norm(RobotAdjusted))
    to_ret_twist = RobotAdjusted + UserAdjusted #linear blend
    if fix_magnitude_user_command:
      to_ret_twist *= np.linalg.norm(self.user_action)/np.linalg.norm(to_ret_twist)
    # dist2goal = np.linalg.norm((self.robot_state["x"])[0:3] - ((self.goals[np.argmax(goal_distribution)]).pos)[0:3])
    # #print("DIST TO GOAL",(dist2goal))
    # sorted = np.sort(goal_distribution)
    # diff = abs(sorted[-1] - sorted[-2])
    # #print("Diff Confidence",diff)
    # blend_mult = (2*diff/.98) + (2*(.52)/((4*dist2goal)+.12)) #.52 is starting distance from every goal
    if np.linalg.norm(to_ret_twist) > .15:
      ratio2 = .14/np.linalg.norm(to_ret_twist) #.14 is avg magnitude of human action
    else:
      ratio2 = 1
    

    return to_ret_twist*ratio2*2
    
      
  def report_assisted_action(self, goal_distribution, fix_magnitude_user_command=False):
    assert goal_distribution.size == len(self.goal_assist_policies)

    action_dimension = GoalPolicy.TargetPolicy.ACTION_DIMENSION
    #TODO how do we handle mode switch vs. not?
    total_action_twist = np.zeros(action_dimension)
    for goal_policy,goal_prob in zip(self.goal_assist_policies, goal_distribution):
      total_action_twist += goal_prob * goal_policy.get_action() #goal policy gets smallest action for the relevant goal given huber parameters, multiplies by probability level for magniutde

    total_action_twist /= np.sum(goal_distribution)
    #user action is 1x7 joint space
    print("ROBOT",np.linalg.norm(total_action_twist))
    print("USER",np.linalg.norm(self.user_action))
    if np.linalg.norm(self.user_action) > .05:
      ratio = np.linalg.norm(self.user_action)/np.linalg.norm(total_action_twist)
      RobotAdjusted = [item * ratio for item in total_action_twist]
      print("ROBOT ADjusted",np.linalg.norm(RobotAdjusted))

      print("#Ratio",ratio)
    else:
      RobotAdjusted = total_action_twist*.125
    #UserAdjusted = [item * 1 for item in self.user_action]
    UserAdjusted = self.user_action
    #RobotAdjusted = [item * ratio for item in total_action_twist]
    
    to_ret_twist = RobotAdjusted + UserAdjusted #linear blend
    #print "before magnitude adjustment: " + str(to_ret_twist)
    if fix_magnitude_user_command:
      to_ret_twist *= np.linalg.norm(self.user_action)/np.linalg.norm(to_ret_twist)
      #print("AHHH BAD COMPUTER")
    #else:
      #print("GOOd comppy",to_ret_twist)
    #print "after magnitude adjustment: " + str(to_ret_twist)
    #print(self.goals)
    dist2goal = np.linalg.norm((self.robot_state["x"])[0:3] - ((self.goals[np.argmax(goal_distribution)]).pos)[0:3])
    #print("DIST TO GOAL",(dist2goal))
    sorted = np.sort(goal_distribution)
    diff = abs(sorted[-1] - sorted[-2])
    #print("Diff Confidence",diff)
    blend_mult = (diff/.98)*((.52)/((4*dist2goal)+.2)) #.52 is starting distance from every goal

    print("DIST TO GOAL",(dist2goal))
    print("Diff Confidence",diff)
    print("Multiplier",blend_mult)
    print("P1--",(diff/.98))
    print("P2--",((.52)/((4*dist2goal)+.2)))
    return 1
  
  # def get_assisted_action(self, goal_distribution, fix_magnitude_user_command=False):
  #   assert goal_distribution.size == len(self.goal_assist_policies)

  #   action_dimension = GoalPolicy.TargetPolicy.ACTION_DIMENSION
  #   #TODO how do we handle mode switch vs. not?
  #   total_action_twist = np.zeros(action_dimension)
  #   for goal_policy,goal_prob in zip(self.goal_assist_policies, goal_distribution):
  #     total_action_twist += goal_prob * goal_policy.get_action() #goal policy gets smallest action for the relevant goal given huber parameters, multiplies by probability level for magniutde

  #   total_action_twist /= np.sum(goal_distribution)
  #   #user action is 1x7 joint space
  #   #print("ROBOT",np.linalg.norm(total_action_twist))
  #   #print("USER",np.linalg.norm(self.user_action))
    
  #   #ratio = np.linalg.norm(self.user_action)/np.linalg.norm(total_action_twist)
  #   #print("#Ratio",ratio)
  #   #UserAdjusted = [item * 1 for item in self.user_action]
  #   UserAdjusted = self.user_action

  #   # dist2goal = np.linalg.norm((self.robot_state["x"])[0:3] - ((self.goals[np.argmax(goal_distribution)]).pos)[0:3])
  #   # #print("DIST TO GOAL",(dist2goal))
  #   # sorted = np.sort(goal_distribution)
  #   # diff = abs(sorted[-1] - sorted[-2])
  #   # #print("Diff Confidence",diff)
  #   # blend_mult = (diff/.98) + ((.52)/((dist2goal)+.52)) #.52 is starting distance from every goal

  #   if np.linalg.norm(self.user_action) > .01:
  #     dist2goal = np.linalg.norm((self.robot_state["x"])[0:3] - ((self.goals[np.argmax(goal_distribution)]).pos)[0:3])
  #     ratio = np.linalg.norm(self.user_action)/np.linalg.norm(total_action_twist)
  #     RobotAdjusted = [item * ratio*((.52)/((dist2goal**.5)+.52)) for item in total_action_twist]
  #     #print("ROBOT ADjusted",np.linalg.norm(RobotAdjusted))
  #     #print("#Ratio",ratio)
  #   else: 
  #     RobotAdjusted = total_action_twist*.75#*blend_mult
  #   #print("ROBOT",np.linalg.norm(RobotAdjusted))
  #   to_ret_twist = RobotAdjusted + UserAdjusted #linear blend
  #   if fix_magnitude_user_command:
  #     to_ret_twist *= np.linalg.norm(self.user_action)/np.linalg.norm(to_ret_twist)
  #   # dist2goal = np.linalg.norm((self.robot_state["x"])[0:3] - ((self.goals[np.argmax(goal_distribution)]).pos)[0:3])
  #   # #print("DIST TO GOAL",(dist2goal))
  #   # sorted = np.sort(goal_distribution)
  #   # diff = abs(sorted[-1] - sorted[-2])
  #   # #print("Diff Confidence",diff)
  #   # blend_mult = (2*diff/.98) + (2*(.52)/((4*dist2goal)+.12)) #.52 is starting distance from every goal
    
    

  #   return to_ret_twist*2
