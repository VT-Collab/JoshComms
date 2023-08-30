import socket
import numpy as np
from AdaHandler2 import *
from AdaAssistancePolicy import goal_from_object

import numpy as np
from Goal import Goal
from Utils import *
from env2 import SimpleEnv

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

# def connect2comms(PORT):
    
#     #s.bind(('192.168.1.26', PORT))
#     s.connect(('172.16.0.3', PORT))
#     conn, addr = s.accept()
#     return conn

def main():


    env = Initialize_Env(visualize=True)

    goals, goal_objects = Initialize_Goals(env, randomize_goal_init=False)
    ada_handler = AdaHandler(env, goals, goal_objects) #goal objects is env objects, goals are GOAL object made from env objects
    
    PORT_robot = 8080

    PORT_comms = 8642

    action_scale = 0.1

    print('[*] Connecting to test comms...')
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect(('172.16.0.3', PORT_comms))
    #conn2 = connect2comms(PORT_comms)
    while True:     
        msg = str(s.recv(2048))[2:-2]
        state_str = list(msg.split(","))
        if msg is not None:
            print("msg",msg)


main()
