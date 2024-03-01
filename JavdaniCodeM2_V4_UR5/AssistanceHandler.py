#The basic ADA interface, such as executing end effector motions, getting the state of the robot, etc.
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
import Tkinter
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


	def execute_policy(self, direct_teleop_only=False, blend_only=True, fix_magnitude_user_command=True, w_comms = True,algo_enabled = False):

		rospy.init_node("teleop")

		mover = TrajectoryClient()
		joystick = Joystick()
		start_time = timestep = timest = time.time()
		rate = rospy.Rate(100)
		motor = False
		pick = False
		pick_count = 0

		print("[*] Initialized, Moving Home")
		mover.switch_controller(mode='position')
		mover.send_joint(HOME, 2.0)
		mover.client.wait_for_result()
		mover.switch_controller(mode='velocity')
		print("[*] Ready for joystick inputs")

		q = mover.samplejoint()
		qclose = q + np.random.normal(0, 0.1, 6)

		
		if (w_comms == True):
			GUI_1 = GUI_Interface()
			GUI_1.root.geometry("+100+100")
			GUI_1.fg = '#ff0000'
			GUI_1.textbox1 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
			GUI_1.textbox1.grid(row = 1, column = 0,  pady = 25, padx = 50) 

			GUI_1.textbox2 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
			GUI_1.textbox2.grid(row = 1, column = 1,   pady = 25, padx = 50) 

			GUI_1.textbox3 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
			GUI_1.textbox3.grid(row = 1, column = 2,   pady = 25, padx = 50) 

			GUI_1.textbox4 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
			GUI_1.textbox4.grid(row = 3, column = 0,   pady = 25, padx = 50)    

			GUI_1.textbox5 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
			GUI_1.textbox5.grid(row = 3, column = 1,   pady = 25, padx = 50) 

			GUI_1.textbox6 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
			GUI_1.textbox6.grid(row = 3, column = 2,  pady = 25, padx = 50) 

			GUI_1.textbox7 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
			GUI_1.textbox7.grid(row = 5, column = 0,   pady = 25, padx = 50)     

			GUI_1.textbox8 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
			GUI_1.textbox8.grid(row = 5, column = 1,   pady = 25, padx = 50) 

			GUI_1.textbox9 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
			GUI_1.textbox9.grid(row = 5, column = 2,  pady = 25, padx = 50) 
			Tracker = "Tired"
			oldmax = 9001
			GUI_1.root.update()

		action_scale = 0.015
		if direct_teleop_only:
			action_scale = .05
		auto_or_noto = True
		autoplus = False
		grasp = True

		#set the huber constants differently if the robot movement magnitude is fixed to user input magnitude
		if not direct_teleop_only and fix_magnitude_user_command:
			for goal_policy in self.robot_policy.assist_policy.goal_assist_policies:
				for target_policy in goal_policy.target_assist_policies:
					target_policy.set_constants(huber_translation_linear_multiplier=1.55, huber_translation_delta_switch=0.11, huber_translation_constant_add=0.2, huber_rotation_linear_multiplier=0.20, huber_rotation_delta_switch=np.pi/72., huber_rotation_constant_add=0.3, huber_rotation_multiplier=0.20, robot_translation_cost_multiplier=14.0, robot_rotation_cost_multiplier=0.05)
					target_policy.set_observation(algo_enabled)

		start_time = time.time()
		sim_time = 0.0
		end_time=time.time()
		left_time = time.time()-2
		right_time = time.time()-2
		b_time = time.time() - 2

		q = mover.joint_states
		x = mover.joint2pose()
		self.robot_state = {"q":q,"x":x}
		
		StateList = []
		UserActionList = []
		AutoActionList = []
		TimeStepList = []
		InputList = []
		
		start = time.time()
		Begin_Switch = False
		print("Waiting on User to Begin")
		while not rospy.is_shutdown():
			z, [A_pressed, B_pressed, X_pressed, Y_pressed],  START,STOP = joystick.getInput()
			if START and (not Begin_Switch):
				Begin_Switch = True
				print("START")
				time.sleep(.5)
				#START = False
				
			if Begin_Switch:

				start = time.time()
				
				xdot = [0]*6
				# if begining:

				# 	time.sleep(.2)
				# 	begining = False
				#get pose of min value target for user's goal
				#self.robot_state = self.env.panda.state
				q = mover.joint_states
				x = mover.joint2pose()
				# J = mover.kdl_kin.jacobian(mover.joint_states)
				if direct_teleop_only:
					print("POS",x[0:3])
				# T = mover.kdl_kin.forward(q)
				self.robot_state = {"q":q,"x":x}
				#print("DEAD")
				z, [A_pressed, B_pressed, X_pressed, Y_pressed],  START,STOP = joystick.getInput()
				#z /=(np.linalg.norm(z)+.1)
				InputList.append(z)
				# if A_pressed:
				# 	xdot[3] = -action_scale * z[0]
				# 	xdot[4] = action_scale * z[1]
				# 	xdot[5] = action_scale * z[2]
				# else:
				xdot[0] = action_scale * z[1]
				xdot[1] = action_scale * z[0]
				xdot[2] = -action_scale * z[2]


				direct_teleop_action = (mover.xdot2qdot(xdot)) #qdot
				#print("USER",np.linalg.norm(direct_teleop_action))

				
				if not direct_teleop_only:
					#When this updates, it updates assist policy and goal policies
					
					goal_distribution = self.robot_policy.goal_predictor.get_distribution()

					max_ind = np.where(goal_distribution == np.max(goal_distribution))[0]
					# maxed = self.goals[max_ind[0]]
					# #print(maxed.name)
					self.robot_policy.update(self.robot_state, direct_teleop_action)
					
					if (w_comms == True and (time.time() - sim_time) > .34): #avoid slow down
						#if confident enough
						sim_time = time.time()
						#print("Comm Runnin",goal_distribution)
						#goal_distribution_sorted = np.sort(goal_distribution)
						max_prob_goal_ind = np.argmax(goal_distribution)
						if self.robot_policy.blend_confidence_function_prob_diff(goal_distribution):
							if oldmax != max_prob_goal_ind:
								s1_time = time.time()
								#GUI_1.fg = '#00ff00'
								#print("AAH", max_prob_goal_ind)
								#print("Ballin")
								if max_prob_goal_ind == 0:
									GUI_1.textbox1 = Entry(GUI_1.root, width = 4, bg = "white", fg='#00ff00', borderwidth = 3, font=("Palatino Linotype", 40))
								else:
									GUI_1.textbox1 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
								GUI_1.textbox1.grid(row = 1, column = 0,  pady = 25, padx = 50)

								if max_prob_goal_ind == 1:
									GUI_1.textbox2 = Entry(GUI_1.root, width = 4, bg = "white", fg='#00ff00', borderwidth = 3, font=("Palatino Linotype", 40))
								else:
									GUI_1.textbox2 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
								GUI_1.textbox2.grid(row = 1, column = 1,  pady = 25, padx = 50)
								
								if max_prob_goal_ind == 2:
									GUI_1.textbox3 = Entry(GUI_1.root, width = 4, bg = "white", fg='#00ff00', borderwidth = 3, font=("Palatino Linotype", 40))
								else:
									GUI_1.textbox3 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
								GUI_1.textbox3.grid(row = 1, column = 2,  pady = 25, padx = 50)



								if max_prob_goal_ind == 3:
									GUI_1.textbox4 = Entry(GUI_1.root, width = 4, bg = "white", fg='#00ff00', borderwidth = 3, font=("Palatino Linotype", 40))
								else:
									GUI_1.textbox4 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
								GUI_1.textbox4.grid(row = 3, column = 0,  pady = 25, padx = 50)

								if max_prob_goal_ind == 4:
									GUI_1.textbox5 = Entry(GUI_1.root, width = 4, bg = "white", fg='#00ff00', borderwidth = 3, font=("Palatino Linotype", 40))
								else:
									GUI_1.textbox5 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
								GUI_1.textbox5.grid(row = 3, column = 1,  pady = 25, padx = 50)

								if max_prob_goal_ind == 5:
									GUI_1.textbox6 = Entry(GUI_1.root, width = 4, bg = "white", fg='#00ff00', borderwidth = 3, font=("Palatino Linotype", 40))
								else:
									GUI_1.textbox6 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
								GUI_1.textbox6.grid(row = 3, column = 2,  pady = 25, padx = 50)





								if max_prob_goal_ind == 6:
									GUI_1.textbox7 = Entry(GUI_1.root, width = 4, bg = "white", fg='#00ff00', borderwidth = 3, font=("Palatino Linotype", 40))
								else:
									GUI_1.textbox7 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
								GUI_1.textbox7.grid(row = 5, column = 0,  pady = 25, padx = 50)

								if max_prob_goal_ind == 7:
									GUI_1.textbox8 = Entry(GUI_1.root, width = 4, bg = "white", fg='#00ff00', borderwidth = 3, font=("Palatino Linotype", 40))
								else:
									GUI_1.textbox8 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
								GUI_1.textbox8.grid(row = 5, column = 1,  pady = 25, padx = 50)

								if max_prob_goal_ind == 8:
									GUI_1.textbox9 = Entry(GUI_1.root, width = 4, bg = "white", fg='#00ff00', borderwidth = 3, font=("Palatino Linotype", 40))
								else:
									GUI_1.textbox9 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
								GUI_1.textbox9.grid(row = 5, column = 2,  pady = 25, padx = 50)
								#print("S1:",time.time()-s1_time)
								oldmax = max_prob_goal_ind
								Tracker = "Goal"
						else:
							#print("Ak",goal_distribution[0])
							if Tracker == "Goal":
								GUI_1.fg = '#ff0000'

								GUI_1.textbox1 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))

								GUI_1.textbox2 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
								
								GUI_1.textbox3 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
								

								GUI_1.textbox4 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
								
								GUI_1.textbox5 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
								
								GUI_1.textbox6 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
								

								GUI_1.textbox7 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
								
								GUI_1.textbox8 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
								
								GUI_1.textbox9 = Entry(GUI_1.root, width = 4, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))

								Tracker = "Tired"
								oldmax = 9001
								
							#print("S2:",time.time()-s2_time)
						#s3_time = time.time()
						
						GUI_1.textbox1.delete(0, END)
						GUI_1.textbox1.grid(row = 1, column = 0,  pady = 25, padx = 50)
						GUI_1.textbox1.insert(0, goal_distribution[0]*100)

						GUI_1.textbox2.delete(0, END)
						GUI_1.textbox2.grid(row = 1, column = 1,  pady = 25, padx = 50)
						GUI_1.textbox2.insert(0, goal_distribution[1]*100)

						GUI_1.textbox3.delete(0, END)
						GUI_1.textbox3.grid(row = 1, column = 2,  pady = 25, padx = 50)
						GUI_1.textbox3.insert(0,goal_distribution[2]*100)

						GUI_1.textbox4.delete(0, END)
						GUI_1.textbox4.grid(row = 3, column = 0,  pady = 25, padx = 50)
						GUI_1.textbox4.insert(0,goal_distribution[3]*100)

						GUI_1.textbox5.delete(0, END)
						GUI_1.textbox5.grid(row = 3, column = 1,  pady = 25, padx = 50)
						GUI_1.textbox5.insert(0,goal_distribution[4]*100)

						GUI_1.textbox6.delete(0, END)
						GUI_1.textbox6.grid(row = 3, column = 2,  pady = 25, padx = 50)
						GUI_1.textbox6.insert(0,goal_distribution[5]*100)
						
						GUI_1.textbox7.delete(0, END)
						GUI_1.textbox7.grid(row = 5, column = 0,  pady = 25, padx = 50)
						GUI_1.textbox7.insert(0,goal_distribution[6]*100)

						GUI_1.textbox8.delete(0, END)
						GUI_1.textbox8.grid(row = 5, column = 1,  pady = 25, padx = 50)
						GUI_1.textbox8.insert(0,goal_distribution[7]*100)

						GUI_1.textbox9.delete(0, END)
						GUI_1.textbox9.grid(row = 5, column = 2,  pady = 25, padx = 50)
						GUI_1.textbox9.insert(0,goal_distribution[8]*100)
						GUI_1.root.update()
						


					if auto_or_noto:
						#action = direct_teleop_action
						action = self.robot_policy.get_blend_action(goal_distribution) #uses in built variables brought by update into maintained class
						
					else:
					
						action = direct_teleop_action
				else:
					action = direct_teleop_action
				
				if X_pressed:
					
					mover.actuate_gripper(0., 0.1, 1)
					print("closed")
					#time.sleep(0.5)
					
				if Y_pressed:
					print("OPEN")
					mover.actuate_gripper(0.25, 0.1, 1)
				#self.env.step(joint = action,mode = 0)
				#Curr_time = time.time()-start
				TimeStepList.append(time.time()-start)
				UserActionList.append(direct_teleop_action)
				
				AutoActionList.append(action)
				StateList.append(self.robot_state)

				if STOP:
					print("[*] Initialized, Moving Home")
					mover.switch_controller(mode='position')
					mover.send_joint(HOME, 4.0)
					mover.client.wait_for_result()
					mover.switch_controller(mode='velocity')
					print("[*] Done! Saving Data!")

					db = {'TotalTime':TimeStepList,'State':StateList,'UserAction':UserActionList,'AutoAction':AutoActionList,'InputList':InputList}
					dbfile = open(self.filename,'ab')
					pickle.dump(db,dbfile)
					dbfile.close()
					print("DATA SAVED")

					
					return True

				if end_time-start_time > 3000.0:
					print("DONE DONE DONE")
					break

	#			if B_pressed and ((end_time-b_time)>.4):


				end_time=time.time()

				#action *= 2*np.linalg.norm(direct_teleop_action)/np.linalg.norm(action)
				mover.sendQ((action))
				#time.sleep(.05)
				mover.client.wait_for_result()

				