import socket
import time
import numpy as np
import pickle
import pygame
import sys


"""
 * a minimal script for teleoperating the robot using a joystick
 * Dylan Losey, September 2020

 * To run:
 [1] in one terminal:
    navigate to ~/panda-ws/essentials
    run python3 teleop.py
 [2] in a second terminal:
    navigate to ~/libfranka/build
    run ./collab/velocity_control
"""


class Joystick(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.deadband = 0.1

    def input(self):
        pygame.event.get()
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
        START_pressed = self.gamepad.get_button(7)
        return [z1, z2, z3], A_pressed, START_pressed


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

def connect2comms(PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #s.bind(('192.168.1.26', PORT))
    s.bind(('172.16.0.3', PORT))
    s.listen()
    conn, addr = s.accept()
    return conn

def send2comms(conn, traj, limit=1.0):
    traj = np.asarray(traj)
    #scale = np.linalg.norm(traj)
    #if scale > limit:
        #traj = np.asarray([traj[i] * limit/scale for i in range(7)])
    #send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
    send_msg = "s," + "insert_data_here" + ","
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
    state["J"] = state_vector[27:].reshape((7,6)).T
    return state

def readState(conn):
    while True:
        state = listen2robot(conn)
        if state is not None:
            break
    return state

def xdot2qdot(xdot, state):
    J_pinv = np.linalg.pinv(state["J"])
    return J_pinv @ np.asarray(xdot)


def main():

    PORT_robot = 8080

    PORT_comms = 8642

    action_scale = 0.1

    print('[*] Connecting to low-level controller...')

    #conn = connect2robot(PORT_robot)

    print('[*] Connecting to test comms...')
    conn2 = connect2comms(PORT_comms)
    send2comms(conn2, [[0,1,2],[1,2,3]])
    print('[*] waiting for test comms...')
    # while True:
    #     message = listen2comms(conn2)
    #     if message is not None:
    #         break
    
    print('[*] Ready for a teleoperation...')   
    interface = Joystick()

    print('[*] Ready for a teleoperation...')

    while True:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        state = readState(conn)
        print(state['q'])

        z, grasp, stop = interface.input()
        print(z)
        if stop:
            print("[*] Done!")
            return True

        xdot = [0]*6
        

        if grasp:
            xdot[3] = -action_scale * z[0]
            xdot[4] = action_scale * z[1]
            xdot[5] = action_scale * z[2]
        else:
            xdot[0] = -action_scale * z[1]
            xdot[1] = -action_scale * z[0]
            xdot[2] = -action_scale * z[2]

        qdot = xdot2qdot(xdot, state)
        send2robot(conn, qdot)
        # time.sleep(1)


if __name__ == "__main__":
    main()
