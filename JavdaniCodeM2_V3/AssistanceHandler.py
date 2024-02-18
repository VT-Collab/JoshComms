#The basic ADA interface, such as executing end effector motions, getting the state of the robot, etc.
from tracemalloc import start
from Primary_AssistPolicy import *
from UserBot import *
#import AssistancePolicyVisualizationTools as vistools
from Utils import *
#from GoalPredictor import *
from GUI import *

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


CONTROL_HZ = 40.

num_control_modes = 2


class AdaHandler:
	def __init__(self, env, goals, goal_objects, goal_object_poses=None,user=44,demo=0):
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
		self.robot_policy = PrimaryPolicy(self.goals)


	
	def execute_policy_simControlled(self, direct_teleop_only=False, blend_only=True, fix_magnitude_user_command=True, w_comms = True,algo_enabled = False):

		self.robot_state = self.panda.state

		if (w_comms == True):	
			GUI_1 = GUI_Interface()
			GUI_1.root.geometry("+100+100")
			GUI_1.fg = '#ff0000'
			GUI_1.textbox1 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
			GUI_1.textbox1.grid(row = 1, column = 0,  pady = 50, padx = 100)     

			GUI_1.textbox2 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
			GUI_1.textbox2.grid(row = 1, column = 1,  pady = 50, padx = 100) 

			GUI_1.textbox3 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
			GUI_1.textbox3.grid(row = 1, column = 2,  pady = 50, padx = 100) 
			Tracker = "Tired"
			oldmax = 9001
			GUI_1.root.update()


		self.interface = Joystick()

		action_scale = 0.25
		
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
		b_time = time.time() - 2

		# StateList = [q]
		# UserActionList = [[0]*7]
		# AutoActionList = [[0]*7]
		SA_time = 0
		
		
			
		begining = True
			
		while True:
			xdot = [0]*6
			# if begining:

			# 	time.sleep(.2)
			# 	begining = False
			#get pose of min value target for user's goal
			self.robot_state = self.env.panda.state
			q = self.robot_state['q']
			x = self.robot_state['x']
			
			z, A_pressed, B_pressed, X_pressed, Y_pressed, START, STOP, RightT, LeftT = self.interface.input()

			# if A_pressed:
			# 	xdot[3] = -action_scale * z[0]
			# 	xdot[4] = action_scale * z[1]
			# 	xdot[5] = action_scale * z[2]
			# else:
			xdot[0] = action_scale * z[1]
			xdot[1] = action_scale * z[0]
			xdot[2] = -action_scale * z[2]

			print(xdot)
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
				goal_distribution = self.robot_policy.goal_predictor.get_distribution()
				self.robot_policy.update(self.robot_state, direct_teleop_action)

				if (w_comms == True and (time.time() - sim_time) > .5): #Update GUI, avoid slow down
					#if confident enough
					sim_time = time.time()
					
					#goal_distribution_sorted = np.sort(goal_distribution)
					max_prob_goal_ind = np.argmax(goal_distribution)
					if self.robot_policy.blend_confidence_function_prob_diff(goal_distribution):
						if oldmax != max_prob_goal_ind:
							s1_time = time.time()
							#GUI_1.fg = '#00ff00'
							if max_prob_goal_ind == 0:
								GUI_1.textbox1 = Entry(GUI_1.root, width = 8, bg = "white", fg='#00ff00', borderwidth = 3, font=("Palatino Linotype", 80))
							else:
								GUI_1.textbox1 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
							GUI_1.textbox1.grid(row = 1, column = 0,  pady = 50, padx = 100)

							if max_prob_goal_ind == 1:
								GUI_1.textbox2 = Entry(GUI_1.root, width = 8, bg = "white", fg='#00ff00', borderwidth = 3, font=("Palatino Linotype", 80))
							else:
								GUI_1.textbox2 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
							GUI_1.textbox2.grid(row = 1, column = 1,   pady = 50, padx = 100)

							if max_prob_goal_ind == 2:
								GUI_1.textbox3 = Entry(GUI_1.root, width = 8, bg = "white", fg='#00ff00', borderwidth = 3, font=("Palatino Linotype", 80))
							else:
								GUI_1.textbox3 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
							GUI_1.textbox3.grid(row = 1, column = 2,   pady = 50, padx = 100)
							
							oldmax = max_prob_goal_ind
							Tracker = "Goal"
					else:
						if Tracker == "Goal":
							GUI_1.fg = '#ff0000'
							GUI_1.textbox1 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
							GUI_1.textbox1.grid(row = 1, column = 0,   pady = 50, padx = 100)     

							GUI_1.textbox2 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
							GUI_1.textbox2.grid(row = 1, column = 1,  pady = 50, padx = 100)

							GUI_1.textbox3 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
							GUI_1.textbox3.grid(row = 1, column = 2,  pady = 50, padx = 100)
							Tracker = "Tired"
							oldmax = 9001
						 	
						
					GUI_1.textbox1.delete(0, END)
					GUI_1.textbox1.insert(0, goal_distribution[0]*100)
				
					GUI_1.textbox2.delete(0, END)
					GUI_1.textbox2.insert(0, goal_distribution[1]*100)

					GUI_1.textbox3.delete(0, END)
					GUI_1.textbox3.insert(0,goal_distribution[2]*100)
					GUI_1.root.update()
					
					

				if auto_or_noto and not direct_teleop_only:
					if blend_only:

						if autoplus:
							action = self.robot_policy.get_blend_action_confident() 
						else:
							action = self.robot_policy.get_blend_action() #uses in built variables brought by update into maintained class
					else:
						action = self.robot_policy.get_action()#see above
				else:
				#if left trigger is being hit, direct teleop
					action = direct_teleop_action
			else:
				action = direct_teleop_action

			print(action)
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
			
				
				return True

			if end_time-start_time > 3000.0:
				print("DONE DONE DONE")
				break

			if B_pressed and ((end_time-b_time)>.4):
				#print("Robot Pos",x)
				v,q = self.robot_policy.assist_policy.get_values()
				action = self.robot_policy.assist_policy.report_assisted_action(goal_distribution)
				for i in self.goals:
					dist = x[0:3]-i.pos
					print(i.name,dist, "Net",np.linalg.norm(dist))
				print("v,q:",v,q)
				b_time = time.time()
				print("Conf",goal_distribution)

			end_time=time.time()
			self.env.step(joint = action,mode = 0, grasp = grasp)


	
	def execute_policy_sim(self, direct_teleop_only=False, blend_only=True, fix_magnitude_user_command=True, w_comms = True,algo_enabled = False):

		self.robot_state = self.panda.state

		if (w_comms == True):	
			GUI_1 = GUI_Interface()
			GUI_1.root.geometry("+100+100")
			GUI_1.fg = '#ff0000'
			GUI_1.textbox1 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
			GUI_1.textbox1.grid(row = 1, column = 0,  pady = 50, padx = 100)     

			GUI_1.textbox2 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
			GUI_1.textbox2.grid(row = 1, column = 1,  pady = 50, padx = 100) 

			GUI_1.textbox3 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
			GUI_1.textbox3.grid(row = 1, column = 2,  pady = 50, padx = 100) 
			Tracker = "Tired"
			oldmax = 9001
			GUI_1.root.update()


		#self.interface = Joystick()

		action_scale = 0.25
		
		auto_or_noto = True
		autoplus = False
		grasp = True

		User = UserBot(self.goals)
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
		b_time = time.time() - 2

		# StateList = [q]
		# UserActionList = [[0]*7]
		# AutoActionList = [[0]*7]
		SA_time = 0
		
		
			
		begining = True
			
		while True:
			xdot = [0]*6
			# if begining:

			# 	time.sleep(.2)
			# 	begining = False
			#get pose of min value target for user's goal
			self.robot_state = self.env.panda.state
			q = self.robot_state['q']
			x = self.robot_state['x']
			
			xdot = User.get_usr_cmd(x[0:3])
			xdot_full = [xdot[0],xdot[1],xdot[2],0,0,0]
			
			direct_teleop_action = xdot2qdot(xdot_full, self.robot_state) #qdot
			
			if not direct_teleop_only:
				#When this updates, it updates assist policy and goal policies
				goal_distribution = self.robot_policy.goal_predictor.get_distribution()
				self.robot_policy.update(self.robot_state, direct_teleop_action)

				if (w_comms == True and (time.time() - sim_time) > .5): #Update GUI, avoid slow down
					#if confident enough
					sim_time = time.time()
					
					#goal_distribution_sorted = np.sort(goal_distribution)
					max_prob_goal_ind = np.argmax(goal_distribution)
					if self.robot_policy.blend_confidence_function_prob_diff(goal_distribution):
						if oldmax != max_prob_goal_ind:
							s1_time = time.time()
							#GUI_1.fg = '#00ff00'
							if max_prob_goal_ind == 0:
								GUI_1.textbox1 = Entry(GUI_1.root, width = 8, bg = "white", fg='#00ff00', borderwidth = 3, font=("Palatino Linotype", 80))
							else:
								GUI_1.textbox1 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
							GUI_1.textbox1.grid(row = 1, column = 0,  pady = 50, padx = 100)

							if max_prob_goal_ind == 1:
								GUI_1.textbox2 = Entry(GUI_1.root, width = 8, bg = "white", fg='#00ff00', borderwidth = 3, font=("Palatino Linotype", 80))
							else:
								GUI_1.textbox2 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
							GUI_1.textbox2.grid(row = 1, column = 1,   pady = 50, padx = 100)

							if max_prob_goal_ind == 2:
								GUI_1.textbox3 = Entry(GUI_1.root, width = 8, bg = "white", fg='#00ff00', borderwidth = 3, font=("Palatino Linotype", 80))
							else:
								GUI_1.textbox3 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
							GUI_1.textbox3.grid(row = 1, column = 2,   pady = 50, padx = 100)
							
							oldmax = max_prob_goal_ind
							Tracker = "Goal"
					else:
						if Tracker == "Goal":
							GUI_1.fg = '#ff0000'
							GUI_1.textbox1 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
							GUI_1.textbox1.grid(row = 1, column = 0,   pady = 50, padx = 100)     

							GUI_1.textbox2 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
							GUI_1.textbox2.grid(row = 1, column = 1,  pady = 50, padx = 100)

							GUI_1.textbox3 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 80))
							GUI_1.textbox3.grid(row = 1, column = 2,  pady = 50, padx = 100)
							Tracker = "Tired"
							oldmax = 9001
						 	
						
					GUI_1.textbox1.delete(0, END)
					GUI_1.textbox1.insert(0, goal_distribution[0]*100)
				
					GUI_1.textbox2.delete(0, END)
					GUI_1.textbox2.insert(0, goal_distribution[1]*100)

					GUI_1.textbox3.delete(0, END)
					GUI_1.textbox3.insert(0,goal_distribution[2]*100)
					GUI_1.root.update()
					
					

				if auto_or_noto and not direct_teleop_only:
					if blend_only:

						if autoplus:
							action = self.robot_policy.get_blend_action_confident() 
						else:
							action = self.robot_policy.get_blend_action() #uses in built variables brought by update into maintained class
							#print("WHIPIT")
					else:
						action = self.robot_policy.get_action()#see above
				else:
				#if left trigger is being hit, direct teleop
					action = direct_teleop_action
			else:
				action = direct_teleop_action

			#	print(action)


			if end_time-start_time > 3000.0:
				print("DONE DONE DONE")
				break


			end_time=time.time()
			self.env.step(joint = action,mode = 0, grasp = grasp)

	