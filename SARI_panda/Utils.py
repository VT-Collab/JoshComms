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

from env2 import SimpleEnv

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
    s.bind(('172.16.0.3', PORT))
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
    # print(state_str)
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
		A_pressed = self.gamepad.get_button(0) and (curr_time - self.lastpress > self.timeband)
		B_pressed = self.gamepad.get_button(1) and (curr_time - self.lastpress > self.timeband)
		X_pressed = self.gamepad.get_button(2) and (curr_time - self.lastpress > self.timeband)
		Y_pressed = self.gamepad.get_button(3) and (curr_time - self.lastpress > self.timeband)
		START_pressed = self.gamepad.get_button(7) and (curr_time - self.lastpress > self.timeband)
		STOP_pressed = self.gamepad.get_button(6) and (curr_time - self.lastpress > self.timeband)
		Right_trigger = self.gamepad.get_button(5)
		Left_Trigger = self.gamepad.get_button(4)
		if A_pressed or START_pressed or B_pressed:
			self.lastpress = curr_time
		return [z1, z2, z3], A_pressed, B_pressed, X_pressed, Y_pressed, START_pressed, STOP_pressed, Right_trigger, Left_Trigger
	
VIEWER_DEFAULT = 'InteractiveMarker'

#def Initialize_Adapy(args, env_path='/environments/tablewithobjects_assisttest.env.xml'):
def Initialize_Env(visualize=False):
    env = SimpleEnv(visualize)
    #Init_Robot(robot)
    
    return env

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
    
    #self.fork_details = {'obj':self.fork,'grasp':self.fork_grasp,'positions':self.fork_poslist,'quats':self.fork_quatlist,'num':len(self.fork_grasp)}
    for i in range(len(env.fork_details['grasp'])):
        fork1 = {'obj':env.fork,'grasp':env.fork_grasp[i],'positions':env.fork_poslist[i],'quats':env.fork_quatlist[i]}
        goal_objects.append(fork1)
    
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
  #print(pose)
  ik_sol = manip._inverse_kinematics(pos, quat)
  target_poses.append(quat)
  target_iks.append(ik_sol)
  #print("ik_sol")
  #print(ik_sol)
  return Goal(quat,pos ,grasp=obj['grasp'], target_poses = target_poses, target_iks = target_iks)


class Goal: 
    
    def __init__(self, pose,pos, grasp,target_poses = list(), target_iks = list()):
      self.pose = pose
      self.quat = pose
      self.goal_num = 0
      self.pos = list(pos)
      self.grasp= grasp
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



# """Obtain the location of target"""
# def get_target():
# 	#x and y adjustment for later use
# 	#Right EE = -.332, .71
# 	#Center -.261 .71
# 	#Left	-.180 .71
# 	x_adjust = -.332
# 	y_adjust = .71

# 	# Configure depth and color streams
# 	pipeline = rs.pipeline()
# 	config = rs.config()
# 	#print("Lights,Camera,Action")
# 	# Get device product line for setting a supporting resolution
# 	pipeline_wrapper = rs.pipeline_wrapper(pipeline)
# 	pipeline_profile = config.resolve(pipeline_wrapper)
# 	device = pipeline_profile.get_device()
# 	device_product_line = str(device.get_info(rs.camera_info.product_line))

# 	found_rgb = False
# 	for s in device.sensors:
# 		if s.get_info(rs.camera_info.name) == 'RGB Camera':
# 			found_rgb = True
# 			break
# 	if not found_rgb:
# 		print("The demo requires Depth camera with Color sensor")
# 		exit(0)

# 	config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# 	config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 	# Start streaming
# 	profile = pipeline.start(config)


# 	# Getting the depth sensor's depth scale (see rs-align example for explanation)
# 	depth_sensor = profile.get_device().first_depth_sensor()
# 	depth_scale = depth_sensor.get_depth_scale()
# 	print("Depth Scale is: " , depth_scale)

# 	# We will be removing the background of objects more than
# 	#  clipping_distance_in_meters meters away
# 	clipping_distance_in_meters = 1 #1 meter
# 	clipping_distance = clipping_distance_in_meters / depth_scale

# 	# Create an align object
# 	# rs.align allows us to perform alignment of depth frames to others frames
# 	# The "align_to" is the stream type to which we plan to align depth frames.
# 	align_to = rs.stream.color
# 	align = rs.align(align_to)

# 	try:
# 		# while True:

# 			# Wait for a coherent pair of frames: depth and color
# 			frames = pipeline.wait_for_frames()

# 			aligned_frames = align.process(frames)

# 			depth_frame = aligned_frames.get_depth_frame()
# 			color_frame = aligned_frames.get_color_frame()
# 			# if not depth_frame or not color_frame:
# 			#     continue

# 			# Convert images to numpy arrays
# 			depth_image = np.asanyarray(depth_frame.get_data())
# 			color_image = np.asanyarray(color_frame.get_data())

# 			gray_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# 			gray_upper = 110
# 			gray_lower = 0

# 			kernal = np.ones ((2, 2), "uint8")

# 			gray1 = cv2.inRange(gray_img, gray_lower, gray_upper)
# 			gray1 = cv2.morphologyEx(gray1, cv2.MORPH_OPEN, kernal)

# 			# FOR CAMERA 1
# 			(contoursred1, hierarchy) =cv2.findContours (gray1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 			for pic, contourred in enumerate (contoursred1):
# 				area = cv2.contourArea (contourred) 
# 				if (area > 0):
# 					x, y, w, h = cv2.boundingRect (contourred)
# 					gray_img = cv2.rectangle (gray_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
# 					cv2.putText(gray_img,"RED",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))

# 			if len(contoursred1) > 0:
# 				# Find the biggest contour
# 				biggest_contour = max(contoursred1, key=cv2.contourArea)

# 				# Find center of contour and draw filled circle
# 				moments = cv2.moments(biggest_contour)
# 				centre_of_contour = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
# 				cv2.circle(gray_img, centre_of_contour, 2, (0, 0, 255), -1)
# 				# Save the center of contour so we draw line tracking it
# 				center_points1 = centre_of_contour
# 				r1 = center_points1[0]
# 				c1 = center_points1[1]

# 				"""
# 				Take the depth as (y,x) when calling it from the image_depth matrix
# 				The x and y axis are flipped
# 				"""

# 				x = r1 - 580
# 				y = 150 - c1
				
# 				# print("The depth at [{},{}] is {}".format(x_adjust-(x/(385*0.435)),y_adjust+(y/(175*0.2)),depth_image[c1, r1]))



# 			# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
# 			depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

# 			depth_colormap_dim = depth_image.shape
# 			color_colormap_dim = color_image.shape

# 			# If depth and color resolutions are different, resize color image to match depth image for display
# 			if depth_colormap_dim != color_colormap_dim:
# 				resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
# 				image_rgb = resized_color_image
# 				image_depth = depth_colormap
		
# 			else:
# 				image_rgb = color_image
# 				image_depth = depth_colormap


# 			# cv2.imshow('RealSense RGB', gray_img)
# 			# cv2.imshow('RealSense Depth', image_depth)
# 			# cv2.waitKey(0)
# 			# print(depth_image[c1, r1])
# 			if depth_image[c1, r1] > 600:
# 				x_adjust = -.325
# 				y_adjust = .71
# 				return x_adjust -(x/395*0.435), y_adjust +(y/172.5*0.2), 0.69-depth_image[c1, r1]/1000+0.035, 'soft'

# 			else:
# 				print("!!!!!!!!!")
# 				x_adjust = -.246
# 				y_adjust =  .71
# 				return x_adjust -(x/395*0.435), y_adjust +(y/172.5*0.2), 0.69-depth_image[c1, r1]/1000 - 0.04, 'rigid'
# 			# if cv2.waitKey(1) & 0xFF == ord('q'):
# 			#     break

# 	finally:

# 		# Stop streaming
# 		pipeline.stop()
