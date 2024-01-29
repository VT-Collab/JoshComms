from operator import truediv
import numpy as np
#import cv2
import time 
import pickle
import socket
import sys
from scipy.interpolate import interp1d
import pygame
#import pyrealsense2 as rs
from tkinter import *
import tf as transmethods

from env3 import SimpleEnv

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

"""Home Position for Panda for all tasks"""
HOME = [0.8385, -0.0609, 0.2447, -1.5657, 0.0089, 1.5335, 1.8607]

  
"""Connecting and Sending commands to robot"""
def connect2robot(PORT):
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  s.bind(('172.16.0.3', PORT))
  s.listen()
  conn, addr = s.accept()
  return conn

def send2robot(conn, qdot, limit=1.0):
    qdot = np.asarray(qdot)
    scale = np.linalg.norm(qdot)
    if scale > limit:
        qdot = np.asarray([qdot[i] * limit/scale for i in range(7)])
    send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
    send_msg = "s," + send_msg + ","
    conn.send(send_msg.encode())

def connect2gripper(PORT):
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  s.bind(('172.16.0.3', PORT))
  s.listen()
  conn, addr = s.accept()
  return conn

def send2gripper(conn, arg):
  # print('-----function called')
  send_msg = arg
  conn.send(send_msg.encode())

def listen2robot(conn):
  state_length = 7 + 7 + 7 + 6 + 42
  message = str(conn.recv(2048))[2:-2]
  state_str = list(message.split(","))
  for idx in range(len(state_str)):
    if state_str[idx] == "s":
      state_str = state_str[idx+1:idx+1+state_length]
      break
  try:
    state_vector = [float(item) for item in state_str]
  except ValueError:
    return None
  if len(state_vector) is not state_length:
    return None
  state_vector = np.asarray(state_vector)
  state = {}
  state["q"] = state_vector[0:7]
  state["dq"] = state_vector[7:14]
  state["tau"] = state_vector[14:21]
  state["O_F"] = state_vector[21:27]
  # print(state_vector[21:27])
  state["J"] = state_vector[27:].reshape((7,6)).T

  # get cartesian pose
  xyz_lin, R = joint2pose(state_vector[0:7])
  beta = -np.arcsin(R[2,0])
  alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
  gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
  xyz_ang = [alpha, beta, gamma]
  xyz = np.asarray(xyz_lin).tolist() + np.asarray(xyz_ang).tolist()
  state["x"] = np.array(xyz)
  return state

def readState(conn):
  while True:
    state = listen2robot(conn)
    if state is not None:
      break
  return state

def connect2comms(PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('127.0.0.1', PORT))
    #s.connect(('192.168.1.3', PORT_comms)) #White Box
    #s.bind(('192.168.1.57', PORT))
    s.listen()
    conn, addr = s.accept()
    return conn

def send2comms(conn, q):

    msg = np.array2string(q, precision=5, separator=',',suppress_small=True)[1:-1]
    send_msg = "s," + msg + ","
    #send_msg = "s," + send_msg + ","
    conn.send(send_msg.encode())

def listen2comms(conn):
    state_length = 7 + 7 + 7 + 6 + 42
    message = str(conn.recv(2048))[2:-2]
    state_str = list(message.split(","))
    print(state_str)
    if message is not None:
        return state_str   
    return None

def xdot2qdot(xdot, state):
  J_pinv = np.linalg.pinv(state["J"])
  xdot[3] = wrap_angles(xdot[3])
  xdot[4] = wrap_angles(xdot[4])
  xdot[5] = wrap_angles(xdot[5])
  
  return J_pinv @ np.asarray(xdot)

def joint2pose(q):
  def RotX(q):
    return np.array([[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])
  def RotZ(q):
    return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
  def TransX(q, x, y, z):
    return np.array([[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])
  def TransZ(q, x, y, z):
    return np.array([[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
  H1 = TransZ(q[0], 0, 0, 0.333)
  H2 = np.dot(RotX(-np.pi/2), RotZ(q[1]))
  H3 = np.dot(TransX(np.pi/2, 0, -0.316, 0), RotZ(q[2]))
  H4 = np.dot(TransX(np.pi/2, 0.0825, 0, 0), RotZ(q[3]))
  H5 = np.dot(TransX(-np.pi/2, -0.0825, 0.384, 0), RotZ(q[4]))
  H6 = np.dot(RotX(np.pi/2), RotZ(q[5]))
  H7 = np.dot(TransX(np.pi/2, 0.088, 0, 0), RotZ(q[6]))
  H_panda_hand = TransZ(-np.pi/4, 0, 0, 0.2105)
  H = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
  return H[:,3][:3], H[:,:][:]

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
    self.deadband = 0.1
    self.timeband = 0.5
    self.lastpress = time.time()

  def input(self):
    pygame.event.get()
    curr_time = time.time()
    z1 = self.gamepad.get_axis(0)
    z2 = self.gamepad.get_axis(1)
    z3 = self.gamepad.get_axis(4)
    if abs(z1) < self.deadband:
      z1 = 0.0
    if abs(z2) < self.deadband:
      z2 = 0.0
    if abs(z3) < self.deadband:
      z3 = 0.0
    A_pressed = self.gamepad.get_button(0) 
    B_pressed = self.gamepad.get_button(1) 
    X_pressed = self.gamepad.get_button(2) 
    Y_pressed = self.gamepad.get_button(3) 
    START_pressed = self.gamepad.get_button(8) 
    STOP_pressed = self.gamepad.get_button(6) 
    Right_trigger = self.gamepad.get_button(5)
    Left_Trigger = self.gamepad.get_button(4)
    if A_pressed or START_pressed or B_pressed:
      self.lastpress = curr_time
    return [z1, z2, z3], A_pressed, B_pressed, X_pressed, Y_pressed, START_pressed, STOP_pressed, Right_trigger, Left_Trigger
     
  def rumble(self,time):
    self.gamepad.rumble(1,1,time)
# class Joystick(object):

#   def __init__(self):
#     pygame.init()
#     self.gamepad = pygame.joystick.Joystick(0)
#     self.gamepad.init()
#     self.deadband = 0.1
#     self.timeband = 0.5
#     self.lastpress = time.time()

#   def input(self):
#     pygame.event.get()
#     curr_time = time.time()
#     z1 = self.gamepad.get_axis(0)
#     z2 = self.gamepad.get_axis(1)
#     z3 = self.gamepad.get_axis(3)
#     if abs(z1) < self.deadband:
#       z1 = 0.0
#     if abs(z2) < self.deadband:
#       z2 = 0.0
#     if abs(z3) < self.deadband:
#       z3 = 0.0
#     A_pressed = self.gamepad.get_button(0) 
#     B_pressed = self.gamepad.get_button(1) 
#     X_pressed = self.gamepad.get_button(2) 
#     Y_pressed = self.gamepad.get_button(3) 
#     START_pressed = self.gamepad.get_button(7) 
#     STOP_pressed = self.gamepad.get_button(6) 
#     Right_trigger = self.gamepad.get_button(5)
#     Left_Trigger = self.gamepad.get_button(4)
#     if A_pressed or START_pressed or B_pressed:
#       self.lastpress = curr_time
#     return [z1, z2, z3], A_pressed, B_pressed, X_pressed, Y_pressed, START_pressed, STOP_pressed, Right_trigger, Left_Trigger
     
#   def rumble(self,time):
#     self.gamepad.rumble(1,1,time)
  
VIEWER_DEFAULT = 'InteractiveMarker'

#def Initialize_Adapy(args, env_path='/environments/tablewithobjects_assisttest.env.xml'):
# def Initialize_Env(visualize=False):
#     env = SimpleEnv(visualize)
#     #Init_Robot(robot)
    
#     return env

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
    # goal_objects.append(env.door)
    
    #fork
    # for i in range(len(env.fork_details['grasp'])):
    #     fork1 = {'obj':env.fork,'grasp':env.fork_grasp[i],'name': env.fork_name[i],'positions':env.fork_poslist[i],'quats':env.fork_quatlist[i]}
    #     goal_objects.append(fork1)
    
    #Cups
    for i in range(len(env.cup1_details['grasp'])):
        cup1 = {'obj':env.cup1,'grasp':env.cup1_grasp[i],'name': env.cup1_name[i],'positions':env.cup1_poslist[i],'quats':env.cup1_quatlist[i]}
        goal_objects.append(cup1)
        print(env.cup1_name[i],env.cup1_poslist[i])
    # for i in range(len(env.cup2_details['grasp'])):
    #     cup2 = {'obj':env.cup2,'grasp':env.cup2_grasp[i],'name': env.cup2_name[i],'positions':env.cup2_poslist[i],'quats':env.cup2_quatlist[i]}
    #     goal_objects.append(cup2)
    # for i in range(len(env.cup3_details['grasp'])):
    #     cup3 = {'obj':env.cup3,'grasp':env.cup3_grasp[i],'name': env.cup3_name[i],'positions':env.cup3_poslist[i],'quats':env.cup3_quatlist[i]}
    #     goal_objects.append(cup3)

    #Mug
    for i in range(len(env.mug_details['grasp'])):
        mug = {'obj':env.mug,'grasp':env.mug_grasp[i],'name': env.mug_name[i],'positions':env.mug_poslist[i],'quats':env.mug_quatlist[i]}
        #print("IM GONNA MUG YOU:", env.mug_quatlist[i])
        goal_objects.append(mug)
        print(env.mug_name[i],env.mug_poslist[i])

    #Salt+Pepper and container
    for i in range(len(env.salt_details['grasp'])):
        salt = {'obj':env.salt,'grasp':env.salt_grasp[i],'name': env.salt_name[i],'positions':env.salt_poslist[i],'quats':env.salt_quatlist[i]}
        goal_objects.append(salt)
        print(env.salt_name[i],env.salt_poslist[i])

    # for i in range(len(env.pepper_details['grasp'])):
    #     pepper = {'obj':env.pepper,'grasp':env.pepper_grasp[i],'name': env.pepper_name[i],'positions':env.pepper_poslist[i],'quats':env.pepper_quatlist[i]}
    #     goal_objects.append(pepper)

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
  for obj in goal_objects:
    goals.append(goal_from_object(env,obj))


  return goals

def goal_from_object(env,obj):
  #pose = object.GetTransform()
  #robot = manip.GetRobot()

  num_poses_desired = 30
  max_num_poses_sample = 500
  object = obj['obj']
 # print(type(object))
  target_poses = []
  target_iks = []
  num_sampled = 0
  manip = env.panda
  pos = obj['positions']
  quat = obj['quats']
  name = obj['name']
  #print(pose)
  #print("OPOS",pos)
  ik_sol = manip._inverse_kinematics(pos, quat)
  target_poses.append(quat)
  target_iks.append(ik_sol)
  #print("ik_sol")
  #print(ik_sol)
  return Goal(quat,pos ,grasp=obj['grasp'], name=name,target_poses = target_poses, target_iks = target_iks)


class Goal: 
    
    def __init__(self, pose,pos, grasp,name,target_poses = list(), target_iks = list()):
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
      #self.compute_quaternions_from_target_poses()

      #print 'NUM POSES: ' + str(len(self.target_poses))

    # def compute_quaternions_from_target_poses(self):
    #   #print(self.target_poses)
    #   self.target_quaternions = [transmethods.quaternion_from_matrix(target_pose) for target_pose in self.target_poses]
    
    def at_goal(self, end_effector_trans):
      for pose,quat in zip(self.target_poses,self.target_quaternions):
        pos_diff =  self.pos - end_effector_trans[0:3,3]
        trans_dist = np.linalg.norm(pos_diff)
        
        #print("Quat",quat)
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
				