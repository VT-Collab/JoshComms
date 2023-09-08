#The basic ADA interface, such as executing end effector motions, getting the state of the robot, etc.
from tracemalloc import start
from Primary_AssistPolicy import *
from UserBot import *
#import AssistancePolicyVisualizationTools as vistools
from Utils import *
#from GoalPredictor import *

#from DataRecordingUtils import TrajectoryData
import pandabase as panda
import numpy as np
import math
import time
import os, sys
#import cPickle as pickle
import argparse
import copy
#import tf
import tf as transmethods
#import rospkg
#import rospy
#import rosli

#import openravepy
#import adapy
#import prpy

#from ada_teleoperation.AdaTeleopHandler import AdaTeleopHandler, Is_Done_Func_Button_Hold
#from ada_teleoperation.RobotState import *




# SIMULATE_DEFAULT = False
# #SIMULATE_VELOCITY_MOVEJOINT = False  #NOTE if true, SIMULATE should also be true
# #FLOATING_HAND_ONLY = False

# RESAVE_GRASP_POSES = True
# cached_data_dir = 'cached_data'

CONTROL_HZ = 40.

num_control_modes = 2


  
class AdaHandler:
	def __init__(self, env, goals, goal_objects, goal_object_poses=None):
#      self.params = {'rand_start_radius':0.04,
#             'noise_pwr': 0.3,  # magnitude of noise
#             'vel_scale': 4.,   # scaling when sending velocity commands to robot
#             'max_alpha': 0.6}  # maximum blending alpha value blended robot policy 
		self.limit_low = [-2.60752, -1.58650, -2.60752, -2.764601, -2.607521, -0.015707, -2.60752,-1,-1]
		self.limit_low = [i * .9 for i in self.limit_low]
		self.limit_high = [2.60752, 1.58650, 2.60752, -0.062831, 2.60752, 3.37721, 2.60752,1.5,1.5]
		self.limit_high = [i * .9 for i in self.limit_high]
		self.env = env
		  #self.robot = robot
		self.goals = goals
		self.PORT_robot = 8080
		self.goal_objects = goal_objects
		if not goal_object_poses and goal_objects:
			self.goal_object_poses = [goal_obj['obj'].get_orientation() for goal_obj in goal_objects]
		else:
			self.goal_object_poses = goal_object_poses
		self.panda = env.panda

		self.robot_policy = AdaAssistancePolicy(self.goals)

	# def joint_limit(self,CURR,ACT):
	# 	use = CURR + ACT
	# 	for i in range(len(use)):
	# 		if use[i] > self.limit_high[i]:
	# 			use[i] = self.limit_high[i] - .02
	# 		if use[i] < self.limit_low[i]: 
	# 			use[i] = self.limit_low[i] + .02
	# 	return use-CURR
	# def pos_limit(self):
	# 	use = self.desired['ee_position']
	# 	for i in range(len(use)):
	# 		if use[i] > self.limit_high[i]:
	# 			use[i] = self.limit_high[i] - .02
	# 		if use[i] < self.limit_low[i]: 
	# 			use[i] = self.limit_low[i] + .02
	# 	self.desired['joint_position'] = use

	def execute_policy(self, direct_teleop_only=False, blend_only=TRUE, fix_magnitude_user_command=TRUE, w_comms = True):
		#goal_distribution = np.array([0.333, 0.333, 0.333])
		print('[*] Connecting to low-level controller...')
		#self.panda = panda()

			
		#vis = vistools.VisualizationHandler()

		#robot_state = self.robot_state
		
		time_per_iter = 1./CONTROL_HZ
		PORT_robot = 8080
					
		conn = connect2robot(PORT_robot)
		PORT_gripper = 8081


		print("[*] Connecting to gripper")
		conn_gripper = connect2gripper(PORT_gripper)
		self.robot_state = readState(conn)
		if (w_comms == True):
			PORT_comms = 8642
			#Inner_port = 8640
			print("connecting to comms")
			conn2 = connect2comms(PORT_comms)

		# start_state = self.robot_state
		# start_pos,start_trans = joint2pose(self.robot_state['q'])


		action_scale = 0.1
		#print(self.robot_state["q"])
		if direct_teleop_only: 
			use_assistance = False
		else:
			use_assistance = True
		auto_or_noto = False

		#set the huber constants differently if the robot movement magnitude is fixed to user input magnitude
		if not direct_teleop_only and fix_magnitude_user_command:
			for goal_policy in self.robot_policy.assist_policy.goal_assist_policies:
				for target_policy in goal_policy.target_assist_policies:
					target_policy.set_constants(huber_translation_linear_multiplier=1.55, huber_translation_delta_switch=0.11, huber_translation_constant_add=0.2, huber_rotation_linear_multiplier=0.20, huber_rotation_delta_switch=np.pi/72., huber_rotation_constant_add=0.3, huber_rotation_multiplier=0.20, robot_translation_cost_multiplier=14.0, robot_rotation_cost_multiplier=0.05)

		start_time = time.time()
		sim_time = 0.0
		end_time=time.time()
		left_time = time.time()-2
		
			
		while True:
			if ((end_time - sim_time) > 3.0 and (w_comms == True)):
				#print("SENT",self.robot_state['q'])
				send2comms(conn2, self.robot_state['q'])
				sim_time = time.time()
			
			xdot = [0]*6
			robot_dof_values = 7
			#get pose of min value target for user's goal
			self.robot_state = readState(conn)
			#ee_pos,ee_trans = joint2pose(self.robot_state['q'])  
			#print("ABAAAAAAAAAAAAAAAAAAAAAAAAAAAA",transmethods.quaternion_from_matrix(ee_trans[0:4,0:4]))
			interface = Joystick()
			z, A_pressed, B_pressed, X_pressed, Y_pressed, START, STOP, RightT, LeftT = interface.input()
			
			#print("buttons",A_pressed, START, B_pressed)

			if A_pressed:
				xdot[3] = -action_scale * z[0]
				xdot[4] = action_scale * z[1]
				xdot[5] = action_scale * z[2]
			else:
				xdot[0] = -action_scale * z[1]
				xdot[1] = -action_scale * z[0]
				xdot[2] = -action_scale * z[2]
			#print(xdot)
			if STOP:
				print("[*] Done!")
				return True
			#print(xdot)
			
			direct_teleop_action = xdot2qdot(xdot, self.robot_state) #qdot
			if LeftT and ((end_time-left_time)>.2):
				left_time = time.time()
				auto_or_noto = not auto_or_noto
				#if left trigger not being hit, then execute with assistance
				print("HELPING = ", auto_or_noto)
			if not direct_teleop_only and auto_or_noto:
				use_assistance = not use_assistance
				#When this updates, it updates assist policy and goal policies
				self.robot_policy.update(self.robot_state, direct_teleop_action)
				#send2comms(conn2, self.robot_state['x'])

				if use_assistance and not direct_teleop_only:
					#action = self.user_input_mapper.input_to_action(user_input_all, robot_state)
					#blend vs normal only dictates goal prediction method and use of confidence screening function to decide whether to act.
					if blend_only:
						action = self.robot_policy.get_blend_action() #uses in built variables brought by update into maintained class
					else:
						action = self.robot_policy.get_action()#see above
				else:
				#if left trigger is being hit, direct teleop
					action = direct_teleop_action
			else:
				#if left trigger is being hit, direct teleop
				action = direct_teleop_action
			#print("DIRECT",direct_teleop_action)
			#print("act",action)
			#action = self.joint_limit(self.robot_state['q'],action)
			#print(action)
			send2robot(conn, action)
			if X_pressed:
				send2gripper(conn_gripper, "c")
				print("closed")
				time.sleep(0.5)

			if Y_pressed:
				print("OPEN")
				send2gripper(conn_gripper, "o")
				time.sleep(0.5)
			#self.env.step(joint = action,mode = 0)
			end_time=time.time()
			if end_time-start_time > 300.0:
				print("DONE DONE DONE")
				break
			
		

	def execute_policy_sim(self, direct_teleop_only=False, blend_only=False, fix_magnitude_user_command=False,  w_comms = True):
		#goal_distribution = np.array([0.333, 0.333, 0.333])
		
		self.user_bot = UserBot(self.goals)
		self.user_bot.set_user_goal(3)
		self.robot_state = self.panda.state
		#self.panda.read_state()
		#self.panda.read_jacobian()

		#vis = vistools.VisualizationHandler()

		#robot_state = self.robot_state
		start_state = self.robot_state
		ee_pos,ee_trans = joint2pose(self.robot_state['q'])
		#time_per_iter = 1./CONTROL_HZ
		#PORT_robot = 8080
		if (w_comms == True):
			PORT_comms = 8642
			#Inner_port = 8640
			print("connecting to comms")
			conn2 = connect2comms(PORT_comms)



		#action_scale = 0.1

		if direct_teleop_only: 
			use_assistance = False
		else:
			use_assistance = True

		#set the huber constants differently if the robot movement magnitude is fixed to user input magnitude
		if not direct_teleop_only and fix_magnitude_user_command:
			for goal_policy in self.robot_policy.assist_policy.goal_assist_policies:
				for target_policy in goal_policy.target_assist_policies:
					target_policy.set_constants(huber_translation_linear_multiplier=1.55, huber_translation_delta_switch=0.11, huber_translation_constant_add=0.2, huber_rotation_linear_multiplier=0.20, huber_rotation_delta_switch=np.pi/72., huber_rotation_constant_add=0.3, huber_rotation_multiplier=0.20, robot_translation_cost_multiplier=14.0, robot_rotation_cost_multiplier=0.05)
		user_goal = self.user_bot.goal_num
		goal_position = self.robot_policy.assist_policy.goal_assist_policies[user_goal].goal.pos
		goal_quat = self.robot_policy.assist_policy.goal_assist_policies[user_goal].goal.quat
		goal_grasp = self.robot_policy.assist_policy.goal_assist_policies[user_goal].goal.grasp
#
		start_time = time.time()
		
		self.robot_state = self.env.panda.state
		sim_time = 0.0
		end_time=time.time()
		
		#print("GOALLL",goal_position,goal_quat)
		#print(goal_position)
		#print("GP")
		#a = 1/0
		#xdes = [0]*6
		#xdes = [goal_position[0],goal_position[1],goal_position[2],start_state['x'] [3],start_state['x'] [4],start_state['x'] [5]]
		
		if goal_grasp == 0:
			grasp = True
		else:
			grasp = False
		
		#self.env.step([0]*7,grasp)
	#print(xdes)
	
		
		while True:
			#TODO: CONFIRM CODE COMPATABILITY WITH BLEND THEN PERFORM REAL WORLD TEST
		
			xdot = [0]*6
			robot_dof_values = 7
			self.robot_state = self.env.panda.state
			if ((end_time - sim_time) > 2.0 and (w_comms == True)):
				#print("SENT",self.robot_state['q'])
				send2comms(conn2, self.robot_state['q'])
				sim_time = time.time()
			#direct_teleop_action = xdot2qdot(xdot, self.env.panda.state) #qdot
			xcurr = self.env.panda.state['ee_position']
			xdot = goal_position - xcurr
			qcurr =  self.env.panda.state['ee_quaternion']
			quat_dot = (goal_quat - qcurr)
			#ee_pos,ee_trans = joint2pose(self.robot_state['q'])  
			#print("CURR:",qcurr,"GOAL",goal_quat)
			direct_teleop_action = self.env.panda._action_finder(mode=1, djoint=[0]*7, dposition=xdot, dquaternion=quat_dot)
			#auto_or_noto = 1
			#print(direct)
			self.robot_policy.update(self.robot_state, direct_teleop_action)
			blend_action = self.robot_policy.get_blend_action() #uses in built variables brought by update into maintained class
			#print(blend_action,"BLEND")
			geet_action = self.robot_policy.get_action(fix_magnitude_user_command=fix_magnitude_user_command)#see above

			#step(self, joint =[0]*7,pos=[0]*3,quat =[0]*4 ,grasp = True,mode = 1):
			#self.env.step(pos=xdot,quat =quat_dot,grasp = grasp,mode = 1)
			self.env.step(joint = geet_action,mode = 0, grasp = grasp)

			end_time=time.time()
			if (np.linalg.norm((goal_position - xcurr))+np.linalg.norm(goal_quat-qcurr)) < .01:
				print("DONE AT GOAL")
				break
			if end_time-start_time > 60.0:
				print("TimeOUt")
				break
		a = self.goals[user_goal].grasp
		#self.goals[user_goal].update()
