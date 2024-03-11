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
		self.n = 4
		self.action_confirmation_constant = 1
		self.action_log_robot = [np.zeros(6)]*self.n
		self.action_log_human = [np.ones(6) ]*self.n 
		self.action_count = 0

		self.action_weight = 6
		self.observ_weight = 1.25
		self.user_norm = .05
		

	
	
	def update(self, robot_state, user_action,max_goal_pos = None,goal_distrib = [],alpha = .4,z= np.zeros([1,6])):
		
		super(HuberAssistancePolicy, self).update(robot_state, user_action)
		
		self.goal_distrib = goal_distrib
		self.user_action = user_action

		#robot_action_list = 
		self.max_ind = np.where(goal_distrib == np.max(goal_distrib))[0]
		if self.observ:
			if self.blend_confidence_function_prob_diff(goal_distribution=goal_distrib,prob_diff_required=.15):
				self.bmax = goal_distrib[self.max_ind[0]]
				self.z = (1-self.bmax)*user_action + self.bmax*(self.get_action())
			else: 
				self.z = user_action
		else:
			#self.bmax = goal_distrib[maxind[0]]
			self.z = user_action


		self.robot_state_after_action = self.state_after_user_action(robot_state, user_action)
		#self.robot_state_after_action2 = self.state_after_user_action(robot_state, self.get_action())
		use = time.time()
		self.position_after_action,self.pose_after_action = self.joint2pose(self.robot_state_after_action)    

		self.dist_translation = np.linalg.norm(np.array(self.robot_state["x"][0:3]) - self.goal_pos[0:3])
		self.dist_translation_aftertrans = np.linalg.norm(np.array(self.position_after_action) - self.goal_pos)
	

		
		if np.linalg.norm(np.array(self.goal_pos) - np.array(max_goal_pos)) < .005:
			print(self.goal.name,self.base_Q_action(),self.base_Q_observ(),self.get_qvalue())
		if self.goal.name == "Mug":
			print(self.goal.name,self.base_Q_action(),self.base_Q_observ(),self.get_qvalue())
		
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

	def get_action(self,goal_pos=None ):
		if goal_pos == None:
			goal_pos = self.goal_pos
		q = self.robot_state["q"]
		
		pose = self.kdl_kin.forward(q)

		pose[:3,3] = (np.reshape(goal_pos,(3,1))+pose[:3,3])/2
	
		goal_q = self.invkin(pose,q)
		qdot =  (goal_q - q) 
		if np.linalg.norm(qdot) > .001:
			qdot *= self.user_norm/(np.linalg.norm(qdot))
		return qdot
	def blend_confidence_function_prob_diff(self,goal_distribution, prob_diff_required=0.25):
		if len(goal_distribution) <= 1:
			print("FAILURE")
			return True

		goal_distribution_sorted = np.sort(goal_distribution)
	 
		return goal_distribution_sorted[-1] - goal_distribution_sorted[-2] > prob_diff_required
	def base_Q(self):
		if self.observ:
			
			bmax = max(self.goal_distrib)
			return ( (self.base_Q_action() ) + ( self.base_Q_observ() ))
		else:
			#
			val = np.linalg.norm(self.user_action)
			
			
			#Total = self.action_confirmation_constant 
	
			return (val)
	
	def base_Q_action(self):
		if self.observ:
			val =( ( (np.linalg.norm(self.user_action))*self.goal_distrib[self.goal_index] )*self.action_weight )
		else:
			val = np.linalg.norm(self.user_action)
		return val 
		
	def base_Q_observ(self):
		if self.observ:
			if np.linalg.norm(self.user_action) < .0005:
				val = 0.0
			
			else:
				val = (np.linalg.norm( (( self.z/np.linalg.norm(self.z) )-(self.z/( np.linalg.norm(self.user_action))))) *self.observ_weight )
		else:
			val = 0.0
		return val


	def get_value(self):
		return (self.get_value_translation() ) 

	def get_qvalue(self):
		return (self.get_qvalue_translation() )*1.25

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
		return  self.get_value_translation(self.dist_translation_aftertrans)

	





	




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
