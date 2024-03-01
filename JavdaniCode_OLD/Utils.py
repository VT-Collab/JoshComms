from operator import truediv
import numpy as np
#import cv2
import time 
import pickle

import sys
from scipy.interpolate import interp1d
import pygame
#import pyrealsense2 as rs
from Tkinter import *
import tf as transmethods

import rospy
import actionlib
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
import copy
from collections import deque

from std_msgs.msg import Float64MultiArray

from controller_manager_msgs.srv import (
    SwitchController, 
    SwitchControllerRequest, 
    SwitchControllerResponse
)

from robotiq_2f_gripper_msgs.msg import (
    CommandRobotiqGripperFeedback, 
    CommandRobotiqGripperResult, 
    CommandRobotiqGripperAction, 
    CommandRobotiqGripperGoal
)

from robotiq_2f_gripper_control.robotiq_2f_gripper_driver import (
    Robotiq2FingerGripperDriver as Robotiq
)
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    GripperCommandAction,
    GripperCommandGoal,
    GripperCommand
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint
)
from sensor_msgs.msg import (
    JointState
)
from geometry_msgs.msg import(
    TwistStamped,
    Twist
)

def RotationMatrixDistance(pose1, pose2):
  quat1 = transmethods.quaternion_from_matrix(pose1)
  quat2 = transmethods.quaternion_from_matrix(pose2)
  return QuaternionDistance(quat1, quat2)


def QuaternionDistance(quat1, quat2):
  quat_between = transmethods.quaternion_multiply(  quat2, transmethods.quaternion_inverse(quat1) )
  return AngleFromQuaternionW(quat_between[-1])

def AngleFromQuaternionW(w):
  w = min(0.9999999, max(-0.999999, w))
  phi = 2.*np.arccos(w)
  return min(phi, 2.* np.pi - phi)


def ApplyTwistToTransform(twist, transform, time=1.):
#  transform[0:3,3] += time * twist[0:3]
#  
#  quat = transmethods.quaternion_from_matrix(transform)
#  quat_after_angular = ApplyAngularVelocityToQuaternion(twist[3:], quat, time)
#  transform[0:3, 0:3] = transmethods.quaternion_matrix(quat_after_angular)[0:3, 0:3]

  transform[0:3,3] += time * twist[0:3]

  angular_velocity = twist[3:]
  angular_velocity_norm = np.linalg.norm(angular_velocity)
  if angular_velocity_norm > 1e-3:
    angle = time*angular_velocity_norm
    axis = angular_velocity/angular_velocity_norm
    transform[0:3,0:3] = np.dot(transmethods.rotation_matrix(angle, axis), transform)[0:3,0:3]

  return transform

def ApplyAngularVelocityToQuaternion(angular_velocity, quat, time=1.):
  angular_velocity_norm = np.linalg.norm(angular_velocity)
  angle = time*angular_velocity_norm
  axis = angular_velocity/angular_velocity_norm

  #angle axis to quaternion formula
  quat_from_velocity = np.append(np.sin(angle/2.)*axis, np.cos(angle/2.))

  return transmethods.quaternion_multiply(quat_from_velocity, quat)

#Collab code

HOME = [-0.03213388124574834, -1.2609770933734339, -1.842959229146139, \
        -np.pi/2, 1.55, 0]



# STEP_SIZE_L = 0.15
STEP_SIZE_L = 0.1
STEP_SIZE_A = 0.2 * np.pi / 4
STEP_TIME = 0.01
DEADBAND = 0.1
MOVING_AVERAGE = 100

robot_urdf = URDF.from_parameter_server()
# def joint2pose(q):
#         base_link = "base_link"
#         end_link = "wrist_3_link"
#         #robot_urdf = URDF.from_parameter_server()
#         kdl_kin = KDLKquetioninematics(robot_urdf, base_link, end_link)

#         state = kdl_kin.forward(q)
#         pos = np.array(state[:3,3]).T
#         pos = pos.squeeze().tolist()
#         R = state[:,:3][:3]
#         euler = transmethods.euler_from_matrix(R)

        # return pos,state
def getJ(robot_state):
    joint_states = robot_state["q"]
    base_link = "base_link"
    end_link = "wrist_3_link"
    
    kdl_kin = KDLKinematics(robot_urdf, base_link, end_link)
    J = kdl_kin.jacobian(joint_states)

def xdot2qdot(xdot,robot_state, J):
        if J == None:
          J =getJ(robot_state)
        J_inv = np.linalg.pinv(J)
        return J_inv.dot(xdot)

def wrap_angles(theta):
  if theta < -np.pi:
    theta += 2*np.pi
  elif theta > np.pi:
    theta -= 2*np.pi
  else:
    theta = theta
  return theta

class Joystick(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.action = None
        self.A_pressed = False
        self.B_pressed = False
        self.X_pressed = False
        self.Y_pressed = False
        self.LT = False
        self.RT = False

    def getInput(self):
        pygame.event.get()
        self.A_pressed = self.gamepad.get_button(0)
        self.B_pressed = self.gamepad.get_button(1)
        self.X_pressed = self.gamepad.get_button(2)
        self.Y_pressed = self.gamepad.get_button(3)
        self.LT = self.gamepad.get_button(4)
        self.RT = self.gamepad.get_button(5)
        return self.getEvent()

    def getEvent(self):
        z1 = self.gamepad.get_axis(0)
        z2 = self.gamepad.get_axis(1)
        z3 = self.gamepad.get_axis(4)
        z = [z1, z2, z3]
        for idx in range(len(z)):
            if abs(z[idx]) < DEADBAND:
                z[idx] = 0.0
        stop = self.gamepad.get_button(6)
        #print(z)
        #stop = False
        start = self.gamepad.get_button(7)
        self.A_pressed = self.gamepad.get_button(0)
        self.B_pressed = self.gamepad.get_button(1)
        self.X_pressed = self.gamepad.get_button(2)
        self.Y_pressed = self.gamepad.get_button(3)
        return z, (self.A_pressed,self.B_pressed,self.X_pressed,self.Y_pressed), start, stop

    def getAction(self, z, jaw = [0,0,0]):
        self.action = (STEP_SIZE_L * -z[1], STEP_SIZE_L * -z[0], STEP_SIZE_L * -z[2], 0, 0, 0)
        #self.gamepad.rumble(1,1,time)

def Initialize_Goals(env,  randomize_goal_init=False):
  while True:
    goals, goal_objects = Init_Goals(env, randomize_goal_init)

    #check if we have sufficient poses for each goal, to see if initialization succeeded
    for goal in goals:
      if len(goal.target_poses) < 25:
        continue
    break

  return goals, goal_objects


def Init_Goals(env, robot, randomize_goal_init=False):
    #TODO: Make skill goal pairs for each of tasks
    goal_objects = []

    
    # goal_objects.append(env.block1)
    # goal_objects.append(env.block2)
    # goal_objects.append(env.block3)
    # goal_objects.append(env.door)\
    #Salt,Pepper,plate,fork,fork2,spoon,spoon2,cup,mug
    #Salt+Pepper and container
    for i in range(len(env.salt_details['grasp'])):
        salt = {'obj':env.salt,'grasp':env.salt_grasp[i],'name': env.salt_name[i],'positions':env.salt_poslist[i],'quats':env.salt_quatlist[i]}
        goal_objects.append(salt)


    for i in range(len(env.pepper_details['grasp'])):
        pepper = {'obj':env.pepper,'grasp':env.pepper_grasp[i],'name': env.pepper_name[i],'positions':env.pepper_poslist[i],'quats':env.pepper_quatlist[i]}
        goal_objects.append(pepper)

    for i in range(len(env.plate_details['grasp'])):
        plate = {'obj':env.plate,'grasp':env.plate_grasp[i],'name': env.plate_name[i],'positions':env.plate_poslist[i],'quats':env.plate_quatlist[i]}
        goal_objects.append(plate)


    #fork
    for i in range(len(env.fork_details['grasp'])):
        fork = {'obj':env.fork,'grasp':env.fork_grasp[i],'name': env.fork_name[i],'positions':env.fork_poslist[i],'quats':env.fork_quatlist[i]}
        goal_objects.append(fork)

    for i in range(len(env.fork2_details['grasp'])):
        fork2 = {'obj':env.fork2,'grasp':env.fork2_grasp[i],'name': env.fork2_name[i],'positions':env.fork2_poslist[i],'quats':env.fork2_quatlist[i]}
        goal_objects.append(fork2)
    #Spoon
    for i in range(len(env.spoon_details['grasp'])):
        Spoon = {'obj':env.spoon,'grasp':env.spoon_grasp[i],'name': env.spoon_name[i],'positions':env.spoon_poslist[i],'quats':env.spoon_quatlist[i]}
        goal_objects.append(Spoon)
    for i in range(len(env.spoon2_details['grasp'])):
        Spoon2 = {'obj':env.spoon2,'grasp':env.spoon2_grasp[i],'name': env.spoon2_name[i],'positions':env.spoon2_poslist[i],'quats':env.spoon2_quatlist[i]}
        goal_objects.append(Spoon2)
    
    #Cups
    for i in range(len(env.cup1_details['grasp'])):
        cup1 = {'obj':env.cup1,'grasp':env.cup1_grasp[i],'name': env.cup1_name[i],'positions':env.cup1_poslist[i],'quats':env.cup1_quatlist[i]}
        goal_objects.append(cup1)
    #Mug
    for i in range(len(env.mug_details['grasp'])):
        mug = {'obj':env.mug,'grasp':env.mug_grasp[i],'name': env.mug_name[i],'positions':env.mug_poslist[i],'quats':env.mug_quatlist[i]}
   
        goal_objects.append(mug)



    # for i in range(len(env.container_details['grasp'])):

    #     container = {'obj':env.container,'grasp':env.container_grasp[i],'name': env.container_name[i],'positions':env.container_poslist[i],'quats':env.container_quatlist[i]}
    #     goal_objects.append(container)

    # if randomize_goal_init:
    #   env.block_position += np.random.rand(3)*0.10 - 0.05
    #   env.reset_box()


    goals = Set_Goals_From_Objects(env,goal_objects)

    #for obj in goal_objects:
    #  obj.Enable(True)

    return goals, goal_objects


def Set_Goals_From_Objects(env,goal_objects):

#    else:
  goals = []
  ind = 0
  for obj in goal_objects:
    goals.append(goal_from_object(env,obj,ind))
    ind +=1


  return goals

def goal_from_object(env,obj,ind):
  #pose = object.GetTransform()
  #robot = manip.GetRobot()

  num_poses_desired = 30
  max_num_poses_sample = 500
  object = obj['obj']

  target_poses = []
  target_iks = []
  num_sampled = 0
  manip = env.panda
  pos = obj['positions']
  quat = obj['quats']
  name = obj['name']

  ik_sol = manip._inverse_kinematics(pos, quat)
  target_poses.append(quat)
  target_iks.append(ik_sol)

  return Goal(quat, pos, grasp=obj['grasp'], index=ind, name=name,target_poses = target_poses, target_iks = target_iks)


class Goal: 
    
    def __init__(self, pose, pos, grasp,index,name,target_poses = list(), target_iks = list()):
      self.pose = pose
      self.quat = pose
      self.goal_num = 0
      self.pos = list(pos)
      self.grasp= grasp
      self.name = name
      if not target_poses:
        target_poses.append(pose)

      #copy the targets
      self.target_poses = list(target_poses)
      self.target_iks = list(target_iks)
      self.target_quaternions = self.quat
      self.ind = index
      #self.compute_quaternions_from_target_poses()

      

    # def compute_quaternions_from_target_poses(self):

    #   self.target_quaternions = [transmethods.quaternion_from_matrix(target_pose) for target_pose in self.target_poses]
    
    def at_goal(self, end_effector_trans):
      for pose,quat in zip(self.target_poses,self.target_quaternions):
        pos_diff =  self.pos - end_effector_trans[0:3,3]
        trans_dist = np.linalg.norm(pos_diff)
        

        quat_dist = QuaternionDistance(transmethods.quaternion_from_matrix(end_effector_trans), quat)

        if (trans_dist < 0.01) and (quat_dist < np.pi/48):
          return True
      # if none of the poses in target_poses returned, then we are not at goal
      return False
    def update(self):
      self.goal_num += 1


class GUI_Interface(object):
	def __init__(self):
		self.root = Tk()
		self.root.geometry("+100+100")
		self.root.title("Uncertainity Output")
		self.update_time = 0.02
		self.fg = '#ff0000'
		font = "Palatino Linotype"

		# X_Y Uncertainty
		self.myLabel1 = Label(self.root, text = "Object", font=(font, 40))
		self.myLabel1.grid(row = 0, column = 0, pady = 50, padx = 50)
		self.textbox1 = Entry(self.root, width = 8, bg = "white", fg=self.fg, borderwidth = 3, font=(font, 40))
		self.textbox1.grid(row = 1, column = 0,  pady = 10, padx = 20)
		self.textbox1.insert(0,0)
	
class TrajectoryClient(object):

    def __init__(self):
        # Action client for joint move commands
        self.client = actionlib.SimpleActionClient(
                '/scaled_pos_joint_traj_controller/follow_joint_trajectory',
                FollowJointTrajectoryAction)
        self.client.wait_for_server()
        # Velocity commands publisher
        self.vel_pub = rospy.Publisher('/joint_group_vel_controller/command',\
                 Float64MultiArray, queue_size=10)
        # Subscribers to update joint state
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_states_cb)
        # service call to switch controllers
        self.switch_controller_cli = rospy.ServiceProxy('/controller_manager/switch_controller',\
                 SwitchController)
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",\
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_states = None
        self.base_link = "base_link"
        self.end_link = "wrist_3_link"
        self.robot_urdf = URDF.from_parameter_server()
        self.kdl_kin = KDLKinematics(self.robot_urdf, self.base_link, self.end_link)
        use = .9
        self.joint_limits_low = [-use*2*np.pi,-use*2*np.pi,-use*2*np.pi,-use*2*np.pi,-use*2*np.pi,-use*2*np.pi]
        self.joint_limits_upper = [use*2*np.pi,use*2*np.pi,use*2*np.pi,use*2*np.pi,use*2*np.pi,use*2*np.pi]

        self.joint_limits_lower = list(self.joint_limits_low)
        self.joint_limits_upper = list(self.joint_limits_upper)

        self.kdl_kin.joint_limits_lower = self.joint_limits_lower
        self.kdl_kin.joint_limits_upper = self.joint_limits_upper
        self.kdl_kin.joint_safety_lower = self.joint_limits_lower
        self.kdl_kin.joint_safety_upper = self.joint_limits_upper
        # Gripper action and client
        action_name = rospy.get_param('~action_name', 'command_robotiq_action')
        self.robotiq_client = actionlib.SimpleActionClient(action_name, \
                                CommandRobotiqGripperAction)
        self.robotiq_client.wait_for_server()
        # Initialize gripper
        goal = CommandRobotiqGripperGoal()
        goal.emergency_release = False
        goal.stop = False
        goal.position = 1.00
        goal.speed = 0.1
        goal.force = 5.0
        # Sends the goal to the gripper.
        self.robotiq_client.send_goal(goal)

        # store previous joint vels for moving avg
        self.qdots = deque(maxlen=MOVING_AVERAGE)
        for idx in range(MOVING_AVERAGE):
            self.qdots.append(np.asarray([0.0] * 6))

    def joint_states_cb(self, msg):
        try:
            if msg is not None:
                states = list(msg.position)
                states[2], states[0] = states[0], states[2]
                self.joint_states = tuple(states) 
        except:
            pass
    
    def joint2pose(self, q=None):
        if q is None:
            q = self.joint_states
        state = self.kdl_kin.forward(q)
        pos = np.array(state[:3,3]).T
        pos = pos.squeeze().tolist()
        R = state[:,:3][:3]
        euler = transmethods.euler_from_matrix(R)

        return pos + list(euler)

    def switch_controller(self, mode=None):
        req = SwitchControllerRequest()
        res = SwitchControllerResponse()

        req.start_asap = False
        req.timeout = 0.0
        if mode == 'velocity':
            req.start_controllers = ['joint_group_vel_controller']
            req.stop_controllers = ['scaled_pos_joint_traj_controller']
            req.strictness = req.STRICT
        elif mode == 'position':
            req.start_controllers = ['scaled_pos_joint_traj_controller']
            req.stop_controllers = ['joint_group_vel_controller']
            req.strictness = req.STRICT
        else:
            rospy.logwarn('Unkown mode for the controller!')

        res = self.switch_controller_cli.call(req)

    def xdot2qdot(self, xdot,q=None):
        if q == None:
            q= self.joint_states
        pose = self.kdl_kin.forward(q)
        pose[:3,3] += np.reshape(xdot,(3,1))

        q2 = self.invkin(pose)
        qdot = q2-q
      
        return qdot
    
    def xdot2qdot(self, xdot,q=None):
        if q == None:
            q= self.joint_states
        
        J = self.kdl_kin.jacobian(q)
        J_inv = np.linalg.pinv(J)
      
        return J_inv.dot(xdot)
  
  

    def send(self, xdot):
        qdot = self.xdot2qdot(xdot)
        self.qdots.append(qdot)
        qdot_mean = np.mean(self.qdots, axis=0).tolist()[0]
        cmd_vel = Float64MultiArray()
        cmd_vel.data = qdot_mean
        self.vel_pub.publish(cmd_vel)

    def sendQ(self, qdot):
        #qdot = self.xdot2qdot(xdot)
        self.qdots.append(qdot)
        qdot_mean = np.mean(self.qdots, axis=0).tolist()[0]
        cmd_vel = Float64MultiArray()
        cmd_vel.data = (qdot_mean)
      
        self.vel_pub.publish(cmd_vel)

    def send_joint(self, pos, time):
        waypoint = JointTrajectoryPoint()
        waypoint.positions = pos
        waypoint.time_from_start = rospy.Duration(time)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names
        goal.trajectory.points.append(waypoint)
        goal.trajectory.header.stamp = rospy.Time.now()
        self.client.send_goal(goal)
        rospy.sleep(time)

    def samplejoint(self):
        q = self.kdl_kin.random_joint_angles()
        return tuple(q)

    def dirkin(self, q):
        pose = np.asarray(self.kdl_kin.forward(q))
        return pose

    def jacobian(self, q):
        return self.kdl_kin.jacobian(q)

    def invkin(self, pose, q=None):
        return self.kdl_kin.inverse(pose, q, maxiter=10000, eps=0.001)

    def invkin_search(self, pose, timeout=1.):
        return self.kdl_kin.inverse_search(pose, timeout)
    def actuate_gripper(self, pos, speed, force):
        Robotiq.goto(self.robotiq_client, pos=pos, speed=speed, force=force, block=True)
        return self.robotiq_client.get_result()