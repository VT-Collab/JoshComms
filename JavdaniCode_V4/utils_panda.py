#!/usr/bin/env python
#import rospy
#import actionlib
import numpy as np
import pygame
import pickle
import socket
import time
#from urdf_parser_py.urdf import URDF
#rom pykdl_utils.kdl_kinematics import KDLKinematics
from collections import deque
#from std_msgs.msg import Float64MultiArray, Float32MultiArray, String


#from waypoints import HOME

# remote teleop imports
# import firebase_admin
# from firebase_admin import db, credentials

#cred = credentials.Certificate('pk.json')
#firebase_admin.initialize_app(cred, {
 #   'databaseURL': 'https://rsa-remote-op-default-rtdb.firebaseio.com/'})
#ref = db.reference('coords/')


HOME = [0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]
STEP_SIZE_L = 0.15
STEP_SIZE_A = 0.2 * np.pi / 4
STEP_TIME = 0.01
DEADBAND = 0.1
MOVING_AVERAGE = 100

class JoystickControl(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.toggle = False
        self.action = None
        self.A_pressed = False
        self.B_pressed = False

    def getInput(self):
        pygame.event.get()
        toggle_angular = self.gamepad.get_button(4)
        toggle_linear = self.gamepad.get_button(5)
        self.A_pressed = self.gamepad.get_button(0)
        self.B_pressed = self.gamepad.get_button(1)
        if not self.toggle and toggle_angular:
            self.toggle = True
        elif self.toggle and toggle_linear:
            self.toggle = False
        return self.getEvent()

    def getEvent(self):
        z1 = self.gamepad.get_axis(0)
        z2 = self.gamepad.get_axis(1)
        z3 = self.gamepad.get_axis(4)
        z = [z1, -z2, -z3]
        for idx in range(len(z)):
            if abs(z[idx]) < DEADBAND:
                z[idx] = 0.0
        stop = self.gamepad.get_button(7)
        X_pressed = self.gamepad.get_button(2)
        B_pressed = self.gamepad.get_button(1)
        A_pressed = self.gamepad.get_button(0)
        return tuple(z), A_pressed, B_pressed, X_pressed, stop

    def getAction(self, z):
        if self.toggle:
            self.action = (0, 0, 0, STEP_SIZE_A * z[0], STEP_SIZE_A * z[1], STEP_SIZE_A * z[2])
        else:
            self.action = (STEP_SIZE_L * z[0], STEP_SIZE_L * z[1], STEP_SIZE_L * z[2], 0, 0, 0)
            
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
        dx = self.gamepad.get_axis(0)
        dy = -self.gamepad.get_axis(1)
        dz = -self.gamepad.get_axis(4)
        if abs(dx) < self.deadband:
            dx = 0.0
        if abs(dy) < self.deadband:
            dy = 0.0
        if abs(dz) < self.deadband:
            dz = 0.0
        A_pressed = self.gamepad.get_button(0) and (curr_time - self.lastpress > self.timeband)
        B_pressed = self.gamepad.get_button(1) and (curr_time - self.lastpress > self.timeband)
        START_pressed = self.gamepad.get_button(7) and (curr_time - self.lastpress > self.timeband)
        if A_pressed or B_pressed or START_pressed:
            self.lastpress = curr_time
        return [dx, dy, dz], A_pressed, B_pressed, START_pressed

class RemoteClient(object):

    def __init__(self):
        self.firebase_sub = rospy.Subscriber('/remote_cmd', Float32MultiArray, self.firebase_cb)
        self.axes = []
        self.mode = 0
        self.slow = 0
        self.grip = 0
        self.start = 0
        self.assist = 0
    
    def firebase_cb(self, msg):
        packet = list(msg.data)
        self.axes = packet[:6]
        self.mode = packet[6]
        self.grip = packet[7]
        self.slow = packet[8]
        self.start = packet[9]
        self.assist = packet[10]
    
    def getInput(self):
        return self.axes, self.grip, self.mode, self.slow, self.start, self.assist

    def print_msg(self):
        print(self.axes, self.grip, self.mode, self.slow, self.start, self.assist)

def connect2robot(PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('172.16.0.3', PORT))
    s.listen()
    conn, addr = s.accept()
    s.setblocking(False)
    return conn

def send2robot(conn, qdot, limit=1.0):
    qdot = np.asarray(qdot)
    scale = np.linalg.norm(qdot)
    if scale > limit:
        qdot = np.asarray([qdot[i] * limit/scale for i in range(7)])
    send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
    send_msg = "s," + send_msg + ","
    conn.send(send_msg.encode())

# def pose2joint(pose, guess=None):
#         if guess is None:
#             guess = self.joint_states
#         return self.kdl_kin.inverse(pose, guess, maxiter=1000)

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
    return H[:,3][:3],H

def xdot2qdot(xdot, state):
    #print(xdot)
    J_pinv = np.linalg.pinv(state["J"])
    return J_pinv @ np.asarray(xdot)

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
        
def go2home(conn):
    home = HOME
    home[0] += np.random.uniform(-np.pi/8, np.pi/8)
    home[1] += np.random.uniform(-np.pi/16, np.pi/16)
    # home[2] += np.random.uniform(-np.pi/16, np.pi/16) 

    total_time = 35.0
    start_time = time.time()
    state = readState(conn)
    current_state = np.asarray(state["q"].tolist())

    # Determine distance between current location and home
    dist = np.linalg.norm(current_state - home)
    curr_time = time.time()
    action_time = time.time()
    elapsed_time = curr_time - start_time

    # If distance is over threshold then find traj home
    while dist > 0.02 and elapsed_time < total_time:
        current_state = np.asarray(state["q"].tolist())

        action_interval = curr_time - action_time
        if action_interval > 0.005:
            # Get human action
            qdot = home - current_state
            qdot = np.clip(qdot, -0.4, 0.4)
            send2robot(conn, qdot.tolist())
            action_time = time.time()

        state = readState(conn)
        dist = np.linalg.norm(current_state - home)
        curr_time = time.time()
        elapsed_time = curr_time - start_time

    # Send completion status
    if dist <= 0.02:
        return True
    elif elapsed_time >= total_time:
        return False
def deform(xi, start, length, tau):
    # length += np.random.choice(np.arange(30, length))
    xi1 = np.asarray(xi).copy()
    A = np.zeros((length+2, length))
    for idx in range(length):
        A[idx, idx] = 1
        A[idx+1,idx] = -2
        A[idx+2,idx] = 1
    R = np.linalg.inv(np.dot(A.T, A))
    U = np.zeros(length)
    gamma = np.zeros((length, 6))
    for idx in range(6):
        U[0] = tau[idx]
        gamma[:,idx] = np.dot(R, U)
    end = min([start+length, xi1.shape[0]-1])
    xi1[start:end+1,:] += gamma[0:end-start+1,:]
    return xi1

def get_rotation_mat(euler):
    
    R_x = np.mat([[1, 0, 0],
                  [0, np.cos(euler[0]), -np.sin(euler[0])],
                  [0, np.sin(euler[0]), np.cos(euler[0])]])

    R_y = np.mat([[np.cos(euler[1]), 0, np.sin(euler[1])],
                  [0, 1, 0],
                  [-np.sin(euler[1]), 0, np.cos(euler[1])]])

    R_z = np.mat([[np.cos(euler[2]), -np.sin(euler[2]), 0],
                  [np.sin(euler[2]), np.cos(euler[2]), 0],
                  [0, 0, 1]])
    R = R_x * R_y * R_z
    #R = R_y
    return R

def convert_to_6d(pos):
    pos_awrap = np.zeros(9)
    pos_awrap[:3] = pos[:3]
    pos_awrap[3:] = get_rotation_mat(pos[3:]).flatten('F')[0,:6]
    return pos_awrap

def get_rotation_mat(euler):
    
    R_x = np.mat([[1, 0, 0],
                  [0, np.cos(euler[0]), -np.sin(euler[0])],
                  [0, np.sin(euler[0]), np.cos(euler[0])]])

    R_y = np.mat([[np.cos(euler[1]), 0, np.sin(euler[1])],
                  [0, 1, 0],
                  [-np.sin(euler[1]), 0, np.cos(euler[1])]])

    R_z = np.mat([[np.cos(euler[2]), -np.sin(euler[2]), 0],
                  [np.sin(euler[2]), np.cos(euler[2]), 0],
                  [0, 0, 1]])
    R = R_x * R_y * R_z
    return R
def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

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
    return H[:,3][:3],H