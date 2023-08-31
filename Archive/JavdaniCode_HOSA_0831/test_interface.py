#The basic ADA interface, such as executing end effector motions, getting the state of the robot, etc.
from tracemalloc import start
from AdaAssistancePolicy import *
from UserBot import *

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
def listen2comms(PORT):
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect(('172.16.0.3', PORT))
    #state_length = 7 + 7 + 7 + 6 + 42
    message = str(s.recv(2048))[2:-2]
    state_str = list(message.split(","))
    #print(state_str)
    send_msg = "s," + "insert_resposne_here" + ","
    #send_msg = "s," + send_msg + ","
    #s.sendall(b"s," + "insert_resposne_here" + ",")
    if message is None:
        return None
    return state_str

def connect2comms(PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #s.bind(('192.168.1.26', PORT))
    s.bind(('192.168.1.57', PORT))
    s.listen()
    conn, addr = s.accept()
    return conn

def send2comms(conn, q, limit=1.0):
    #
    #scale = np.linalg.norm(traj)
    #if scale > limit:
        #traj = np.asarray([traj[i] * limit/scale for i in range(7)])
    #print("DOOOOOOOOOOT",q)
    msg = np.array2string(q, precision=5, separator=',',suppress_small=True)[1:-1]
    send_msg = "s," + msg + ","
    #send_msg = "s," + send_msg + ","
    conn.send(send_msg.encode())

#
def main():
    Port_comms = 8642
    print("Connecting to comms")
    conn = connect2comms(Port_comms)

    b = time.time()
    a = time.time()
    c = time.time()+7
    print("Fucking Running")
    while((b-a)<30):
        a = time.time()
        q = np.array([0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0])
        if abs(c-a)>7:
            send2comms(conn, q)
            c = time.time()

main()
        
