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

	def __init__(self, goals,algo):
		self.goals = goals
		
		self.assist_policy = SecondaryPolicy(goals,algo)
		self.goal_predictor = GoalPredictor.GoalPredictor(goals)
		self.alpha = 0.4
		self.usernorm = .065

		self.robot_action =np.zeros([1,6])

	def update(self, robot_state, user_action):
		
		goal_distribution = self.goal_predictor.get_distribution()
		
		self.assist_policy.update(robot_state, user_action,goal_distribution,self.alpha)
		self.user_action = user_action
		self.max_ind = np.where(goal_distribution == np.max(goal_distribution))[0]
		self.max_goal = self.goals[self.max_ind[0]]

		BaseQValues,values,qvalues = self.assist_policy.get_values()
		self.goal_predictor.update_distribution(BaseQValues,qvalues,values)
		
		self.robot_state = robot_state

		dist2goal = np.linalg.norm(np.array(self.robot_state["x"][0:3]) - np.array(self.max_goal.pos[0:3]))
		if dist2goal > .05:
			Robot_action = self.assist_policy.goal_assist_policies[self.max_ind[0]].get_direct_action(self.max_goal.pos)
			self.robot_action = np.reshape(self.usernorm*Robot_action/np.linalg.norm(Robot_action),(1,6))
		else:
			self.robot_action = np.zeros([1,6])
		#print("ues",np.linalg.norm(user_action),np.linalg.norm(self.robot_action))
		

		


	#uses diffent goal distribution
	def get_blend_action(self, goal_distribution = np.array([]), **kwargs):
		
		#Robot_action = self.assist_policy.Robot_action

		#usernorm = 
	
		if self.blend_confidence_function_prob_diff(goal_distribution):
			if np.linalg.norm(self.user_action)< .0005:
				#print("MOVIN",self.alpha,np.linalg.norm(self.robot_action))
				self.alpha += .03
			else:
				#print("kinda",self.alpha,np.linalg.norm(self.robot_action))
				self.alpha -= .04
			if self.alpha > 1.0:
				self.alpha = 1.0
			if self.alpha < .3:
				self.alpha = .3
		else:
			#print("Meh")
			self.alpha = .25
		#print("alpha",self.alpha_belief , self.alpha_trust ,self.alpha_align,self. alpha)
		#print(self. alpha_align)
		#print("True Alpha",self.alpha)
		

		assisted_qdot = self.assist_policy.get_assisted_action(goal_distribution, robot_action=self.robot_action,alpha = self.alpha,**kwargs)

		return assisted_qdot





	def blend_confidence_function_prob_diff(self,goal_distribution, prob_diff_required=0.15):
		if len(goal_distribution) <= 1:
			print("FAILURE")
			return True

		goal_distribution_sorted = np.sort(goal_distribution)
	 
		return goal_distribution_sorted[-1] - goal_distribution_sorted[-2] > prob_diff_required


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

