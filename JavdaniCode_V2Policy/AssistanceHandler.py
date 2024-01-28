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
	def __init__(self, env, goals, goal_objects, goal_object_poses=None,user=44,demo=0):
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
		self.filename = "data/user" + str(user) + "/demo" +str(demo)+ ".pkl"
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

	def execute_policy(self, direct_teleop_only=False, blend_only=TRUE, fix_magnitude_user_command=TRUE, w_comms = True,confVibe= True):
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
		q = self.robot_state['q']

		if (w_comms == True):
			PORT_comms = 8642
			#Inner_port = 8640
			print("connecting to comms")
			conn2 = connect2comms(PORT_comms)

		# start_state = self.robot_state
		# start_pos,start_trans = joint2pose(self.robot_state['q'])

		# GUI_1 = GUI_Interface()
		# GUI_1.root.geometry("+100+100")
		# GUI_1.myLabel1 = Label(GUI_1.root, text = "Confidence", font=("Palatino Linotype", 40))
		# GUI_1.myLabel1.grid(row = 0, column = 0, pady = 50, padx = 50)
		# GUI_1.root.update()
		self.interface = Joystick()

		action_scale = 0.025
		r_action_scale = action_scale*5
		#print(self.robot_state["q"])
		if direct_teleop_only: 
			use_assistance = False
		else:
			use_assistance = True
		auto_or_noto = False
		autoplus = False

		#set the huber constants differently if the robot movement magnitude is fixed to user input magnitude
		if not direct_teleop_only and fix_magnitude_user_command:
			for goal_policy in self.robot_policy.assist_policy.goal_assist_policies:
				for target_policy in goal_policy.target_assist_policies:
					target_policy.set_constants(huber_translation_linear_multiplier=1.55, huber_translation_delta_switch=0.11, huber_translation_constant_add=0.2, huber_rotation_linear_multiplier=0.20, huber_rotation_delta_switch=np.pi/72., huber_rotation_constant_add=0.3, huber_rotation_multiplier=0.20, robot_translation_cost_multiplier=14.0, robot_rotation_cost_multiplier=0.05)

		start_time = time.time()
		sim_time = 0.0
		end_time=time.time()
		left_time = time.time()-2
		right_time = time.time()-2

		StateList = [q]
		UserActionList = [[0]*7]
		AutoActionList = [[0]*7]
		SA_time = 0
		
		
			
		while True:
			xdot = [0]*6
			robot_dof_values = 7
			#get pose of min value target for user's goal
			self.robot_state = readState(conn)
			q = self.robot_state['q']

			
			z, A_pressed, B_pressed, X_pressed, Y_pressed, START, STOP, RightT, LeftT = self.interface.input()
			#print(X_pressed)
			if A_pressed:
				xdot[3] = -action_scale * z[0]
				xdot[4] = action_scale * z[1]
				xdot[5] = action_scale * z[2]
			else:
				xdot[0] = -action_scale * z[1]
				xdot[1] = -action_scale * z[0]
				xdot[2] = -action_scale * z[2]



			
			direct_teleop_action = xdot2qdot(xdot, self.robot_state) #qdot
			if LeftT and ((end_time-left_time)>.4):
				left_time = time.time()
				auto_or_noto = not auto_or_noto
				#if left trigger not being hit, then execute with assistance
				print("HELPING = ", auto_or_noto)
			if RightT and ((end_time-right_time)>.4):
				right_time = time.time()
				autoplus = not autoplus
				print("HELPING Plus = ", autoplus)
			
			if not direct_teleop_only:
				
				#When this updates, it updates assist policy and goal policies
				self.robot_policy.update(self.robot_state, direct_teleop_action)
				
				goal_distribution = self.robot_policy.goal_predictor.get_distribution()
				max_prob_goal_ind = np.argmax(goal_distribution)
				print(max_prob_goal_ind,"TEEEST")
				log_goal_distribution = self.robot_policy.goal_predictor.log_goal_distribution
				#print(goal_distribution)
				# goals = self.goals
				# curr_goal = goals[max_prob_goal_ind]
				# name = curr_goal.name

				#print(goal_distribution[max_prob_goal_ind])

				if ((end_time - sim_time) > 2.0 and (w_comms == True)):
					use = np.append(q,log_goal_distribution)
					#print("SENT",self.robot_state['q'])
					#print("SENT",np.shape(log_goal_distribution))
					send2comms(conn2, use)
					sim_time = time.time()
					
				#print(name,goal_distribution)
				if confVibe:
					#print()
					if self.robot_policy.blend_confidence_function_prob_diff(goal_distribution):
						self.interface.rumble(200)

				if auto_or_noto and not direct_teleop_only:
					#action = self.user_input_mapper.input_to_action(user_input_all, robot_state)
					#blend vs normal only dictates goal prediction method and use of confidence screening function to decide whether to act.
					SA_time += time.time()-end_time
					if blend_only:
						#print("AUTOPLUS:",autoplus)
						if autoplus:
							action = self.robot_policy.get_blend_action_confident() 
						else:
							action = self.robot_policy.get_blend_action() #uses in built variables brought by update into maintained class
					else:
						action = self.robot_policy.get_action()*r_action_scale #see above
				else:
				#if left trigger is being hit, direct teleop
					action = direct_teleop_action
			else:
				action = direct_teleop_action

			send2robot(conn, action)
			end_time=time.time()

			StateList.append(q)
			UserActionList.append(direct_teleop_action)
			AutoActionList.append(action)


			if X_pressed:
				send2gripper(conn_gripper, "c")
				print("closed")
				time.sleep(0.5)

			if Y_pressed:
				print("OPEN")
				send2gripper(conn_gripper, "o")
				time.sleep(0.5)
			#self.env.step(joint = action,mode = 0)
			if STOP:
				print("[*] Done!")
				db = {'TotalTime':abs(time.time-end_time), 'SA_time':SA_time,'State':StateList,'UserAction':UserActionList,'AutoAction':AutoActionList}
				dbfile = open(self.filename,'ab')
				pickle.dump(db,dbfile)
				dbfile.close()
				print("DATA SAVED")
				return True

			if end_time-start_time > 300.0:
				print("DONE DONE DONE")
				break
			
		

	def execute_policy_sim(self, direct_teleop_only=False, blend_only=False, fix_magnitude_user_command=False,  w_comms = True):
		#goal_distribution = np.array([0.333, 0.333, 0.333])
		
		self.user_bot = UserBot(self.goals)
		self.user_bot.set_user_goal(0)
		self.robot_state = self.panda.state

		start_state = self.robot_state
		ee_pos,ee_trans = joint2pose(self.robot_state['q'])

		if (w_comms == True):
			PORT_comms = 8642
			print("connecting to comms")
			conn2 = connect2comms(PORT_comms)

		if direct_teleop_only: 
			use_assistance = False
		else:
			use_assistance = True

		#set the huber constants differently if the robot movement magnitude is fixed to user input magnitude
		if not direct_teleop_only and fix_magnitude_user_command:
			for goal_policy in self.robot_policy.assist_policy.goal_assist_policies:
				for target_policy in goal_policy.target_assist_policies:
					target_policy.set_constants(huber_translation_linear_multiplier=1.55, huber_translation_delta_switch=0.11, huber_translation_constant_add=0.2, huber_rotation_linear_multiplier=0.20, huber_rotation_delta_switch=np.pi/72., huber_rotation_constant_add=0.3, huber_rotation_multiplier=0.20, robot_translation_cost_multiplier=14.0, robot_rotation_cost_multiplier=0.05)
#		
		user_goal = self.user_bot.goal_num
		goal_position = self.robot_policy.assist_policy.goal_assist_policies[user_goal].goal.pos
		goal_quat = self.robot_policy.assist_policy.goal_assist_policies[user_goal].goal.quat
		goal_grasp = self.robot_policy.assist_policy.goal_assist_policies[user_goal].goal.grasp
#
		self.robot_state = self.env.panda.state
#
		start_time = time.time()
		sim_time = 0.0
		end_time=time.time()
#		
		if goal_grasp == 0:
			grasp = True
		else:
			grasp = False
#		
		while True:		
			xdot = [0]*6
			robot_dof_values = 7
			self.robot_state = self.env.panda.state
			log_goal_distribution = self.robot_policy.goal_predictor.log_goal_distribution
#			
			if ((end_time - sim_time) > 0.05 and (w_comms == True)):
			#if ( (w_comms == True)):
				print("LOG",log_goal_distribution)
				use = np.append(self.robot_state['q'],log_goal_distribution)
				send2comms(conn2, use)
				sim_time = time.time()
			#direct_teleop_action = xdot2qdot(xdot, self.env.panda.state) #qdot
#
			xcurr = self.env.panda.state['ee_position']
			xdot = goal_position - xcurr
			qcurr =  self.env.panda.state['ee_quaternion']
			quat_dot = (goal_quat - qcurr)
			direct_teleop_action = self.env.panda._action_finder(mode=1, djoint=[0]*7, dposition=xdot, dquaternion=quat_dot)
			self.robot_policy.update(self.robot_state, direct_teleop_action)
			blend_action = self.robot_policy.get_blend_action(goal_distribution=self.robot_policy.goal_predictor.log_goal_distribution) #uses in built variables brought by update into maintained class
			#geet_action = self.robot_policy.get_action(fix_magnitude_user_command=fix_magnitude_user_command)#see above
			
#
			#print(blend_action,"BLEND")
			
			self.env.step(joint = blend_action,mode = 0, grasp = grasp)
			end_time=time.time()
			#print(end_time-start_time)
#
			if (np.linalg.norm((goal_position - xcurr))+np.linalg.norm(goal_quat-qcurr)) < .1:
				print("DONE AT GOAL")
				break
			if end_time-start_time > 60.0:
				print("TimeOUt")
				break
#
		#a = self.goals[user_goal].grasp

	def execute_policy_simControlled(self, direct_teleop_only=False, blend_only=TRUE, fix_magnitude_user_command=TRUE, w_comms = True,confVibe= True):

		self.robot_state = self.panda.state

		start_state = self.robot_state
		ee_pos,ee_trans = joint2pose(self.robot_state['q'])

		if (w_comms == True):
			PORT_comms = 8642
			#Inner_port = 8640
			print("connecting to comms")
			conn2 = connect2comms(PORT_comms)


		self.interface = Joystick()

		action_scale = 0.125
		r_action_scale = action_scale*5
		#print(self.robot_state["q"])
		if direct_teleop_only: 
			use_assistance = False
		else:
			use_assistance = True
		auto_or_noto = False
		autoplus = False
		grasp = True

		#set the huber constants differently if the robot movement magnitude is fixed to user input magnitude
		if not direct_teleop_only and fix_magnitude_user_command:
			for goal_policy in self.robot_policy.assist_policy.goal_assist_policies:
				for target_policy in goal_policy.target_assist_policies:
					target_policy.set_constants(huber_translation_linear_multiplier=1.55, huber_translation_delta_switch=0.11, huber_translation_constant_add=0.2, huber_rotation_linear_multiplier=0.20, huber_rotation_delta_switch=np.pi/72., huber_rotation_constant_add=0.3, huber_rotation_multiplier=0.20, robot_translation_cost_multiplier=14.0, robot_rotation_cost_multiplier=0.05)

		start_time = time.time()
		sim_time = 0.0
		end_time=time.time()
		left_time = time.time()-2
		right_time = time.time()-2

		# StateList = [q]
		# UserActionList = [[0]*7]
		# AutoActionList = [[0]*7]
		SA_time = 0
		
		
			
		while True:
			xdot = [0]*6
			robot_dof_values = 7
			#get pose of min value target for user's goal
			self.robot_state = self.panda.state
			q = self.robot_state['q']
			ee_pos,ee_trans = joint2pose(self.robot_state['q'])

			
			z, A_pressed, B_pressed, X_pressed, Y_pressed, START, STOP, RightT, LeftT = self.interface.input()
			print(z)
			if A_pressed:
				xdot[3] = -action_scale * z[0]
				xdot[4] = action_scale * z[1]
				xdot[5] = action_scale * z[2]
			else:
				xdot[0] = -action_scale * z[1]
				xdot[1] = -action_scale * z[0]
				xdot[2] = -action_scale * z[2]



			
			direct_teleop_action = xdot2qdot(xdot, self.robot_state) #qdot
			if LeftT and ((end_time-left_time)>.4):
				left_time = time.time()
				auto_or_noto = not auto_or_noto
				#if left trigger not being hit, then execute with assistance
				print("HELPING = ", auto_or_noto)
			if RightT and ((end_time-right_time)>.4):
				right_time = time.time()
				autoplus = not autoplus
				print("HELPING Plus = ", autoplus)
			
			if not direct_teleop_only:
				
				#When this updates, it updates assist policy and goal policies
				self.robot_policy.update(self.robot_state, direct_teleop_action)

				if ((end_time - sim_time) > 2.0 and (w_comms == True)):
					goal_distribution = self.robot_policy.goal_predictor.get_distribution()
					max_prob_goal_ind = np.argmax(goal_distribution)
					print(max_prob_goal_ind,"TEEEST")
					log_goal_distribution = self.robot_policy.goal_predictor.log_goal_distribution
					use = np.append(q,log_goal_distribution)
					send2comms(conn2, use)
					sim_time = time.time()

				if auto_or_noto and not direct_teleop_only:
					#action = self.user_input_mapper.input_to_action(user_input_all, robot_state)
					#blend vs normal only dictates goal prediction method and use of confidence screening function to decide whether to act.
					SA_time += time.time()-end_time
					if blend_only:
						#print("AUTOPLUS:",autoplus)
						if autoplus:
							action = self.robot_policy.get_blend_action_confident() 
						else:
							action = self.robot_policy.get_blend_action() #uses in built variables brought by update into maintained class
					else:
						action = self.robot_policy.get_action()*r_action_scale #see above
				else:
				#if left trigger is being hit, direct teleop
					action = direct_teleop_action
			else:
				action = direct_teleop_action

			#send2robot(conn, action)
			
			# StateList.append(q)
			# UserActionList.append(direct_teleop_action)
			# AutoActionList.append(action)


			if X_pressed:
				#send2gripper(conn_gripper, "c")
				grasp = False
				print("closed")
				time.sleep(0.5)
				

			if Y_pressed:
				print("OPEN")
				#send2gripper(conn_gripper, "o")
				grasp = True
				time.sleep(0.5)
			#self.env.step(joint = action,mode = 0)
			if STOP:
				print("[*] Done!")
				# db = {'TotalTime':abs(time.time-end_time), 'SA_time':SA_time,'State':StateList,'UserAction':UserActionList,'AutoAction':AutoActionList}
				# dbfile = open(self.filename,'ab')
				# pickle.dump(db,dbfile)
				# dbfile.close()
				# print("DATA SAVED")
				return True

			if end_time-start_time > 50.0:
				print("DONE DONE DONE")
				break

			end_time=time.time()
			self.env.step(joint = action,mode = 0, grasp = grasp)