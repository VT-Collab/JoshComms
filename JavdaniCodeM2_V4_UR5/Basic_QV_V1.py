import numpy as np
#import IPython
import AssistancePolicyOneTarget
#from Utils import *
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
import time
import tf as transmethods

ACTION_DIMENSION = 6

class HuberAssistancePolicy(AssistancePolicyOneTarget.AssistancePolicyOneTarget):
	def __init__(self, goal):
		super(HuberAssistancePolicy, self).__init__(goal)
		self.set_constants(self.TRANSLATION_LINEAR_MULTIPLIER, self.TRANSLATION_DELTA_SWITCH, self.TRANSLATION_CONSTANT_ADD, self.ROTATION_LINEAR_MULTIPLIER,
												self.ROTATION_DELTA_SWITCH, self.ROTATION_CONSTANT_ADD, self.ROTATION_MULTIPLIER)
		self.base_link = "base_link"
		self.end_link = "wrist_3_link"
		robot_urdf = URDF.from_parameter_server()
		self.kdl_kin = KDLKinematics(robot_urdf, self.base_link, self.end_link)
		use = .9
		self.joint_limits_low = [-use*2*np.pi,-use*2*np.pi,-use*2*np.pi,-use*2*np.pi,-use*2*np.pi,-use*2*np.pi]
		self.joint_limits_upper = [use*2*np.pi,use*2*np.pi,use*2*np.pi,use*2*np.pi,use*2*np.pi,use*2*np.pi]

		self.joint_limits_lower = list(self.joint_limits_low)
		self.joint_limits_upper = list(self.joint_limits_upper)

		self.kdl_kin.joint_limits_lower = self.joint_limits_lower
		self.kdl_kin.joint_limits_upper = self.joint_limits_upper
		self.kdl_kin.joint_safety_lower = self.joint_limits_lower
		self.kdl_kin.joint_safety_upper = self.joint_limits_upper
	
		self.goal = goal
		self.goal_pos = self.goal.pos
		self.goal_quat = self.goal.quat
		self.goal_index = self.goal.ind
		self.observ = False
		self.n = 15
		self.action_confirmation_constant = 1
		self.action_log_robot = [np.zeros(6)]*self.n
		self.action_log_human = [np.ones(6) ]*self.n 
		self.action_count = 0
		
	def joint2pose(self,q):
		state = self.kdl_kin.forward(q)
		pos = np.array(state[:3,3]).T
		pos = pos.squeeze().tolist()
		#R = state[:,:3][:3]
		return pos,state
	def xdot2qdot(self, xdot,q):
				J = self.kdl_kin.jacobian(q)
				J_inv = np.linalg.pinv(J)
			 
				return J_inv.dot(xdot)
	def invkin(self, pose, q=None):
				return self.kdl_kin.inverse(pose, q, maxiter=10000, eps=0.01)
	
	
	def update(self, robot_state, user_action,max_goal_pos = None,goal_distrib = []):
		
		super(HuberAssistancePolicy, self).update(robot_state, user_action)
		
		self.belief = goal_distrib
		self.user_action = user_action

		b = goal_distrib[self.goal_index]
		# if self.observ:
		#   z = (1-b)*user_action + b*self.get_action()
		# else:
		#   z = user_action
		z = user_action
		
		self.robot_state_after_action = self.state_after_user_action(robot_state, z)
		#self.robot_state_after_action2 = self.state_after_user_action(robot_state, self.get_action())
		use = time.time()
		self.position_after_action,self.pose_after_action = self.joint2pose(self.robot_state_after_action)    

		self.dist_translation = np.linalg.norm(np.array(self.robot_state["x"][0:3]) - self.goal_pos)
		self.dist_translation_aftertrans = np.linalg.norm(np.array(self.position_after_action) - self.goal_pos)

		vec2goal = self.get_action()
		vecnorm = np.linalg.norm(vec2goal)
		if vecnorm<.001:
			vecnorm =1
		self.action_log_robot[self.action_count] = vec2goal / vecnorm
		#print("USE IN CHECKIN",self.action_log_robot)
		
		if np.linalg.norm(self.user_action) < .05:
			#print("SLOW")
			#print(np.linalg.norm(self.user_action))
			vec2goal = self.get_action(goal_pos=max_goal_pos)
			self.action_log_human[self.action_count] = vec2goal / vecnorm
		else:
			self.action_log_human[self.action_count] = self.user_action / np.linalg.norm(self.user_action)
			
		diff = 0
		for i in range(self.n):

			diff += np.linalg.norm(self.action_log_human[i] - self.action_log_robot[i]) / self.n
		self.action_confirmation_constant = diff    
		self.action_count +=1
		if self.action_count == self.n:
			self.action_count = 0
			#print("TESTING --",self.goal.name,"---",self.action_confirmation_constant,self.get_qvalue_translation(),self.action_confirmation_constant+self.get_qvalue_translation())
			
		
		#something about mode switch distance???

	def get_action(self,goal_pos=None ):
		if goal_pos == None:
			goal_pos = self.goal_pos
		q = self.robot_state["q"]
		#goal_euler= np.array(transmethods.euler_from_quaternion(self.goal_quat))
		#goal_x = np.append(self.goal_pos,[0,0,0])
		pose = self.kdl_kin.forward(q)
		#print(pose)
		pose[:3,3] = (np.reshape(goal_pos,(3,1))+pose[:3,3])/2
		#print(pose)
		goal_q = self.invkin(pose,q)
		qdot =  (goal_q - q) 
		if np.linalg.norm(qdot) > .001:
			qdot *= .1/(np.linalg.norm(qdot))
		#print("SHAPIN UP",np.shape(xdot))
		#robot_qdot= self.xdot2qdot(xdot, self.robot_state["q"]) #qdot
	 # print("SHAPIN UP",np.shape(robot_qdot[0,0:6]))
		 #= .001*(robotq[:7] - current_q[:7])
		return qdot
	
	def base_Q(self,q2a = 2):
		if self.observ:
			#print("WRONG")
			#return (self.get_qvalue_translation() + np.linalg.norm(self.user_action))
			return (self.get_qvalue_translation())
		else:
			#
			qval = self.get_qvalue_translation()
			
			Total = qval + self.action_confirmation_constant 
	
			return (qval + self.action_confirmation_constant )
	



	def get_value(self):
		return (self.get_value_translation() )*self.V_Change_Constant

	def get_qvalue(self):
		return (self.get_qvalue_translation() )*self.Q_Change_Constant

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
	
	def set_observation(self,algo=True):
		self.observ = algo

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
