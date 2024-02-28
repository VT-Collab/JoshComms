#Generic assistance policy for one goal
import numpy as np
import time
#import IPython
import AssistancePolicyOneGoal as GoalPolicy

class SecondaryPolicy:

	def __init__(self, goals):
		self.goals = goals
		#self.panda = panda

		self.goal_assist_policies = []
		for goal in goals:
			self.goal_assist_policies.append(GoalPolicy.AssistancePolicyOneGoal(goal))

		#self.user_input_mapper = UserInputMapper()


	def update(self, robot_state, user_action,goal_distrib = []):
		self.robot_state = robot_state
		#user action corresponds to the effect of direct teleoperation on the robot
		#self.user_action = self.user_input_mapper.input_to_action(user_input, robot_state)
		self.user_action = user_action
		self.max_ind = np.where(goal_distrib == np.max(goal_distrib))[0]
		self.max_goal = self.goals[self.max_ind[0]]
		self.Robot_action = np.zeros(6)
		for goal_policy in self.goal_assist_policies:
			goal_policy.update(robot_state, self.user_action,self.max_goal.pos,goal_distrib,self.Robot_action)

		dist2goal = np.linalg.norm(np.array(self.robot_state["x"][0:2]) - np.array(self.max_goal.pos[0:2]))
		if dist2goal > .05:			
			Robot_action = self.goal_assist_policies[self.max_ind[0]].get_action()/2
			self.Robot_action = np.reshape(Robot_action,(1,6))
		else:
			height_diff = self.robot_state["x"][2] - self.max_goal.pos[2]
			if height_diff > .1:				
				Robot_action = self.goal_assist_policies[self.max_ind[0]].get_action()/2
				self.Robot_action = np.reshape(Robot_action,(1,6))
			else:
				self.Robot_action = np.zeros([1,6])
		#max_goal_pos = self.max_goal.pos


	def get_values(self):
		values = np.ndarray(len(self.goal_assist_policies))
		# qvalues = np.ndarray(len(self.goal_assist_policies))
		BaseQvalues = np.ndarray(len(self.goal_assist_policies))
		for ind,goal_policy in enumerate(self.goal_assist_policies):
			BaseQvalues[ind] = goal_policy.get_BaseQValue()
			values[ind] = goal_policy.get_value()
			#print(values[ind]-qvalues[ind])
		#time.sleep(20)

		return BaseQvalues,values



	def get_assisted_action(self, goal_distribution, fix_magnitude_user_command=False,algo_enabled = False,alpha = 0.1):
		assert goal_distribution.size == len(self.goal_assist_policies)

		# action_dimension = GoalPolicy.TargetPolicy.ACTION_DIMENSION
		
		# # Robot_action = np.zeros(action_dimension)
		# # for goal_policy,goal_prob in zip(self.goal_assist_policies, goal_distribution):
		# # 	Robot_action += goal_prob * goal_policy.get_action()
		# # 	#print(a,Robot_action)
		# # 	# for i in range(action_dimension): #goal policy gets smallest action for the relevant goal given huber parameters, multiplies by probability level for magniutde
		# # 	#   print(i,a,a[0,i])
		# # 	#   Robot_action[i] += a[0,i]
		## Robot_action /= len(goal_distribution)
		#max_prob_goal_ind = np.argmax(goal_distribution)
		#print("HEre",self.max_goal.pos)
		Robot_action = self.Robot_action
	
		# if np.linalg.norm(self.user_action) < .05:
		# 	UserAdjusted= Robot_action
		# else:
		UserAdjusted = self.user_action
		z = UserAdjusted #user action
		#print(alpha)
		
		#print("Robot_action",Robot_action,(np.reshape(Robot_action,(1,6))))
		to_ret_twist = ((1-alpha)*z + alpha*Robot_action) #linear blend

		blend_curve = .5 + (2.5*alpha)
		#print("1--Action",np.linalg.norm(to_ret_twist),np.linalg.norm(self.user_action),blend_curve)
		if fix_magnitude_user_command:
			to_ret_twist *= ( .1/np.linalg.norm(to_ret_twist) )*blend_curve
		else:
			to_ret_twist *= blend_curve
		#print("Action",np.linalg.norm(to_ret_twist),np.linalg.norm(self.user_action))
		return to_ret_twist

 
	# def get_assisted_action(self, goal_distribution, fix_magnitude_user_command=False,algo_enabled = False,alpha = .5):
	# 	assert goal_distribution.size == len(self.goal_assist_policies)

	# 	action_dimension = GoalPolicy.TargetPolicy.ACTION_DIMENSION
	# 	#TODO how do we handle mode switch vs. not?
	# 	Robot_action = np.zeros(action_dimension)
	# 	for goal_policy,goal_prob in zip(self.goal_assist_policies, goal_distribution):
	# 		Robot_action += goal_prob * goal_policy.get_action()
	# 		#print(a,Robot_action)
	# 		# for i in range(action_dimension): #goal policy gets smallest action for the relevant goal given huber parameters, multiplies by probability level for magniutde
	# 		#   print(i,a,a[0,i])
	# 		#   Robot_action[i] += a[0,i]
	# 	Robot_action /= len(goal_distribution)

	# 	UserAdjusted = self.user_action

	# 	sorted_goals = np.sort(goal_distribution)
	# 	b = sorted_goals[-1]- sorted_goals[-2] 
	# 	# if algo_enabled:
	# 	#   z = (1-b)*UserAdjusted + b*Robot_action
	# 	# else:
	# 	z = UserAdjusted

	# 	to_ret_twist = (1-alpha)*z + alpha*Robot_action #linear blend
		
	# 	if fix_magnitude_user_command:
	# 		to_ret_twist *= (np.linalg.norm(self.user_action)/np.linalg.norm(to_ret_twist))*b*2

	# 	return to_ret_twist
		