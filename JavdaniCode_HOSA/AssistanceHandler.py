#The basic ADA interface, such as executing end effector motions, getting the state of the robot, etc.
from tracemalloc import start
from Primary_AssistPolicy import *
from UserBot import *
#import AssistancePolicyVisualizationTools as vistools
from Utils import *
#from GoalPredictor import *

#from DataRecordingUtils import TrajectoryData

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
      #self.manip = self.robot.arm
    
      #self.ada_teleop = AdaTeleopHandler(env, robot, input_interface_name, num_input_dofs, use_finger_mode)#, is_done_func=Teleop_Done)

      

        self.robot_policy = AdaAssistancePolicy(self.goals)
      #Ada policy has the following important features
        #get action and get blended action
        #self.assist_policy and self.goal_predictor
        #blend_confidence_function_prob_diff
        #blend_confidence_function_euclidean_distance
        #goal_from_object(env,object)

      #self.user_input_mapper = self.ada_teleop.user_input_mapper

#   def GetEndEffectorTransform(self):
# #    if FLOATING_HAND_ONLY:
# #      return self.floating_hand.GetTransform()
# #    else:
#       return self.manip.GetEndEffectorTransform()


  #def Get_Robot_Policy_Action(self, goal_distribution):
  #    end_effector_trans = self.GetEndEffectorTransform()
  #    return self.robot_policy.get_action(goal_distribution, end_effector_trans)

    def execute_policy(self, direct_teleop_only=False, blend_only=False, fix_magnitude_user_command=False, w_comms = True):
        #goal_distribution = np.array([0.333, 0.333, 0.333])
        print('[*] Connecting to low-level controller...')

            
        #vis = vistools.VisualizationHandler()

        #robot_state = self.robot_state
        
        time_per_iter = 1./CONTROL_HZ
        PORT_robot = 8080
                    
        conn = connect2robot(PORT_robot)
        self.robot_state = readState(conn)
        if (w_comms == True):
            PORT_comms = 8642
            #Inner_port = 8640
            print("connecting to comms")
            conn2 = connect2comms(PORT_comms)

        # start_state = self.robot_state
        # start_pos,start_trans = joint2pose(self.robot_state['q'])


        action_scale = 0.1

        if direct_teleop_only: 
            use_assistance = False
        else:
            use_assistance = True

        #set the huber constants differently if the robot movement magnitude is fixed to user input magnitude
        if not direct_teleop_only and fix_magnitude_user_command:
            for goal_policy in self.robot_policy.assist_policy.goal_assist_policies:
                for target_policy in goal_policy.target_assist_policies:
                    target_policy.set_constants(huber_translation_linear_multiplier=1.55, huber_translation_delta_switch=0.11, huber_translation_constant_add=0.2, huber_rotation_linear_multiplier=0.20, huber_rotation_delta_switch=np.pi/72., huber_rotation_constant_add=0.3, huber_rotation_multiplier=0.20, robot_translation_cost_multiplier=14.0, robot_rotation_cost_multiplier=0.05)

        start_time = time.time()

        
            
        while True:
            if ((end_time - sim_time) > 7.0 and (w_comms == True)):
                #print("SENT",self.robot_state['q'])
                send2comms(conn2, self.robot_state['q'])
                sim_time = time.time()
            
            xdot = [0]*6
            robot_dof_values = 7
            #get pose of min value target for user's goal
            self.robot_state = readState(conn)
            ee_pos,ee_trans = joint2pose(self.robot_state['q'])  
            interface = Joystick()
            z, A_pressed, B_pressed, X_pressed, Y_pressed, START, STOP, RightT, LeftT = interface.input()
            
            

            if A_pressed:
                xdot[3] = -action_scale * z[0]
                xdot[4] = action_scale * z[1]
                xdot[5] = action_scale * z[2]
            else:
                xdot[0] = -action_scale * z[1]
                xdot[1] = -action_scale * z[0]
                xdot[2] = -action_scale * z[2]

            if STOP:
                print("[*] Done!")
                return True

            
            direct_teleop_action = xdot2qdot(xdot, self.env.panda.state) #qdot
            auto_or_noto = LeftT
                #if left trigger not being hit, then execute with assistance
            if not direct_teleop_only and auto_or_noto:
                use_assistance = not use_assistance
                #When this updates, it updates assist policy and goal policies
                self.robot_policy.update(self.robot_state, direct_teleop_action,self.panda)
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

            send2robot(conn, action)
            
            end_time=time.time()
            if end_time-start_time > 300.0:
                print("DONE DONE DONE")
                break
            
        

    def execute_policy_sim(self, direct_teleop_only=False, blend_only=False, fix_magnitude_user_command=False,  w_comms = True):
        #goal_distribution = np.array([0.333, 0.333, 0.333])
        
        self.user_bot = UserBot(self.goals)
        self.user_bot.set_user_goal(0)
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
            if ((end_time - sim_time) > 7.0 and (w_comms == True)):
                #print("SENT",self.robot_state['q'])
                send2comms(conn2, self.robot_state['q'])
                sim_time = time.time()
            #direct_teleop_action = xdot2qdot(xdot, self.env.panda.state) #qdot
            xcurr = self.env.panda.state['ee_position']
            xdot = goal_position - xcurr
            qcurr =  self.env.panda.state['ee_quaternion']
            quat_dot = goal_quat - qcurr
            
            direct_teleop_action = self.env.panda._action_finder(mode=1, djoint=[0]*7, dposition=xdot, dquaternion=quat_dot)
            #auto_or_noto = 1
            #print(direct)
            self.robot_policy.update(self.robot_state, direct_teleop_action,self.panda)
            blend_action = self.robot_policy.get_blend_action() #uses in built variables brought by update into maintained class
            #print(blend_action,"BLEND")
            geet_action = self.robot_policy.get_action(fix_magnitude_user_command=fix_magnitude_user_command)#see above

            #step(self, joint =[0]*7,pos=[0]*3,quat =[0]*4 ,grasp = True,mode = 1):
            #self.env.step(pos=xdot,quat =quat_dot,grasp = grasp,mode = 1)
            self.env.step(joint = blend_action,mode = 0, grasp = grasp)

            end_time=time.time()
            if (np.linalg.norm((goal_position - xcurr))+np.linalg.norm(goal_quat-qcurr)) < .005:
                print("DONE DONE DONE")
                break
            if end_time-start_time > 60.0:
                print("TimeOUt")
                break
        a = self.goals[user_goal].grasp
        #self.goals[user_goal].update()
