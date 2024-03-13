#Generic assistance policy for one goal
import numpy as np
import time
#import IPython
import AssistancePolicyOneGoal as GoalPolicy

class SecondaryPolicy:

	def __init__(self, goals,algo):
		self.goals = goals
		#self.panda = panda
		self.algo = algo
		#self.Robot_action = np.zeros([1,6])

		self.goal_assist_policies = []
		self.user_norm = .03
		for goal in goals:
			self.goal_assist_policies.append(GoalPolicy.AssistancePolicyOneGoal(goal))
		self.z = np.zeros([1,6])
		#self.user_input_mapper = UserInputMapper()


	def update(self, robot_state, user_action,goal_distrib = [],alpha = .25):
		self.robot_state = robot_state
		#user action corresponds to the effect of direct teleoperation on the robot
		#self.user_action = self.user_input_mapper.input_to_action(user_input, robot_state)
		self.user_action = user_action

		self.max_ind = np.where(goal_distrib == np.max(goal_distrib))[0]
		self.max_goal = self.goals[self.max_ind[0]]
		#print("maxind",self.max_ind,goal_distrib)
		for goal_policy in self.goal_assist_policies:
			goal_policy.update(robot_state, self.user_action,self.max_goal.pos,goal_distrib,alpha)

		
		#self.Robot_action= np.zeros(6)

		
	

		
		#max_goal_pos = self.max_goal.pos


	def get_values(self):
		values = np.ndarray(len(self.goal_assist_policies))
		qvalues = np.ndarray(len(self.goal_assist_policies))
		BaseQvalues = np.ndarray(len(self.goal_assist_policies))
		for ind,goal_policy in enumerate(self.goal_assist_policies):
			BaseQvalues[ind] = goal_policy.get_BaseQValue()
			values[ind] = goal_policy.get_value()
			qvalues[ind] = goal_policy.get_qvalue()
			
		#time.sleep(20)

		return BaseQvalues,values,qvalues



	def get_assisted_action(self, goal_distribution, robot_action,fix_magnitude_user_command=False,algo_enabled = False,alpha = 0.1):
		assert goal_distribution.size == len(self.goal_assist_policies)


		Robot_action = robot_action
		if np.linalg.norm(self.user_action) < .0005:			
			self.user_norm = .06
			#self.user_action = Robot_action
			#print("INACTIVE")
		else:
			self.user_norm = (np.linalg.norm(self.user_action)+.01)
			#print("ACTIVE")

		UserAdjusted = self.user_action 
		
		#print(alpha)
		
		if np.linalg.norm(Robot_action) < .0005:
			to_ret_twist = self.user_action
			blend_curve =  1
		else:
			to_ret_twist = ((1-alpha)*UserAdjusted + alpha*Robot_action) #linear blend
			blend_curve = 1.5 + (1.5*alpha)

		if fix_magnitude_user_command:
			to_ret_twist *= ( self.user_norm )*blend_curve
		else:
			to_ret_twist *= blend_curve
		#print("Action",(to_ret_twist),np.linalg.norm(self.user_action))
		return to_ret_twist

