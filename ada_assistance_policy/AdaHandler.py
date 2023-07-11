#The basic ADA interface, such as executing end effector motions, getting the state of the robot, etc.
from tracemalloc import start
from AdaAssistancePolicy import *
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
#import roslib

#import openravepy
#import adapy
#import prpy

#from ada_teleoperation.AdaTeleopHandler import AdaTeleopHandler, Is_Done_Func_Button_Hold
from teleop_comms_Test import readState
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
        self.goal_object_poses = [goal_obj.get_orientation() for goal_obj in goal_objects]
      else:
        self.goal_object_poses = goal_object_poses

      self.sim = env.panda
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

  def execute_policy(self, simulate_user=False, direct_teleop_only=False, blend_only=False, fix_magnitude_user_command=False,  finish_trial_func=None, traj_data_recording=None):
      #goal_distribution = np.array([0.333, 0.333, 0.333])
      if simulate_user:
        self.user_bot = UserBot(self.goals)
        self.user_bot.set_user_goal(0)
        self.robot_state = self.sim._read_state2()
        
      else:
        print('[*] Connecting to low-level controller...')
        self.robot_state = readState(self.PORT_robot)
        conn = connect2robot(PORT_robot)
        
      #vis = vistools.VisualizationHandler()

      robot_state = self.robot_state
      
      ee_pos,ee_trans = joint2pose(robot_state['q'])
      time_per_iter = 1./CONTROL_HZ
      PORT_robot = 8080

      #PORT_comms = 8642


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


      #if specified traj data for recording, initialize
      if traj_data_recording:
        assist_type = 'shared_auton'
        if direct_teleop_only:
          assist_type = 'None'
        elif blend_only:
          assist_type = 'blend'
        elif fix_magnitude_user_command:
          assist_type = 'shared_auton_prop'
        
        traj_data_recording.set_init_info(start_state=copy.deepcopy(robot_state), goals=copy.deepcopy(self.goals), input_interface_name=self.ada_teleop.teleop_interface, assist_type=assist_type)

      while True:
        start_time = time.time()
        xdot = [0]*6
        robot_dof_values = 7
        if simulate_user:
          self.robot_state = self.sim._read_state2()
          #get pose of min value target for user's goal
          user_goal = self.user_bot.goal_num
          ee_pos,ee_trans = joint2pose(robot_state['q'])
          min_val_target_pose = self.robot_policy.assist_policy.goal_assist_policies[user_goal].goal.pos
          user_input_velocity = self.user_bot.get_usr_cmd(ee_pos, goal_pos=min_val_target_pose)
          #user_input_all = UserInputData(user_input_velocity)
          xdot [:3]= user_input_velocity
          #user_input_all.switch_assistance_val
          
          if self.goals[user_goal].at_goal(ee_trans):
            A_pressed = 1
          else:
            A_pressed = 0
        else:
          robot_state = readState(self.PORT_robot)
          ee_pos,ee_trans = joint2pose(robot_state['q'])  
          print('[*] Ready for a teleoperation...')
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

        
        #print user_input_all
#          user_input_velocity = user_input_all.move_velocity
#          user_input_closehand = user_input_all.close_hand_velocity
        direct_teleop_action = xdot2qdot(xdot, robot_state) #qdot
        if simulate_user:
          auto_or_noto = 1
        else:
          auto_or_noto = LeftT
        #if left trigger not being hit, then execute with assistance
        if not direct_teleop_only and auto_or_noto:
          use_assistance = not use_assistance
        #When this updates, it updates assist policy and goal policies
        self.robot_policy.update(robot_state, direct_teleop_action)
        if use_assistance and not direct_teleop_only:
          #action = self.user_input_mapper.input_to_action(user_input_all, robot_state)
          #blend vs normal only dictates goal prediction method and use of confidence screening function to decide whether to act.
          if blend_only:
            action = self.robot_policy.get_blend_action() #uses in built variables brought by update into maintained class
          else:
            action = self.robot_policy.get_action(fix_magnitude_user_command=fix_magnitude_user_command)#see above
        else:
          #if left trigger is being hit, direct teleop
          action = direct_teleop_action

        #qdot = xdot2qdot(action, robot_state)
        if simulate_user:
          self.env.step(action)
        else:
          send2robot(conn, action)

       

#        for goal,goal_obj in zip(self.goals, self.goal_objects):
#          marker_ns = goal_obj.GetName() + '_targets'
#          vis.draw_hand_poses(goal.target_poses, marker_ns=marker_ns)

        #vis.draw_hand_poses([self.GetEndEffectorTransform()], marker_ns='ee_axis')

        #vis.draw_action_arrows(ee_trans, direct_teleop_action.twist[0:3], action.twist[0:3]-direct_teleop_action.twist[0:3])

        ### end visualization ###

        end_time=time.time()

        if traj_data_recording:
          traj_data_recording.add_datapoint(robot_state=copy.deepcopy(robot_state), robot_dof_values=copy.copy(robot_dof_values), user_input_all=copy.deepcopy(user_input_all), direct_teleop_action=copy.deepcopy(direct_teleop_action), executed_action=copy.deepcopy(action), goal_distribution=self.robot_policy.goal_predictor.get_distribution())



        #print ('time: %.5f' % (end_time-start_time)) + '   per iter: ' + str(time_per_iter)
        #print 'sleep time: ' + str(max(0., time_per_iter - (end_time-start_time)))

        #rospy.sleep( max(0., time_per_iter - (end_time-start_time)))

        #if (max(action.finger_vel) > 0.5):
        if end_time-start_time > 3.0:
          print("potato")
          break

      #set the intended goal and write data to file
      if traj_data_recording:
        values, qvalues = self.robot_policy.assist_policy.get_values()
        traj_data_recording.set_end_info(intended_goal_ind=np.argmin(values))
        traj_data_recording.tofile()

      #execute zero velocity to stop movement
      #self.ada_teleop.execute_joint_velocities(np.zeros(len(self.manip.GetDOFValues())))


      # if finish_trial_func:     
      #   finish_trial_func()

 

