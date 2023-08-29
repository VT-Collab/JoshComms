import socket
import numpy as np
import time

from AdaHandler2 import *

from Utils import *
from AdaAssistancePolicy import *
from UserBot import *

def listen2comms(PORT):
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #s.connect(('192.168.1.26', PORT))
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
    s.bind(('172.16.0.3', PORT))
    s.listen()
    conn, addr = s.accept()
    return conn

def send2comms(conn, msg, limit=1.0):
    #
    #scale = np.linalg.norm(traj)
    #if scale > limit:
        #traj = np.asarray([traj[i] * limit/scale for i in range(7)])
    #send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
    send_msg = "s," + msg + ","
    #send_msg = "s," + send_msg + ","
    conn.send(send_msg.encode())


def main():

    print("initializing test environment")
    env = Initialize_Env(visualize=True)

    #initialize HOSA goals and handler
    #set the huber constants differently if the robot movement magnitude is fixed to user input magnitude
    goals, goal_objects = Initialize_Goals(env, randomize_goal_init=False)
    ada_handler = AdaHandler(env, goals, goal_objects) #goal objects is env objects, goals are GOAL object made from env objects
    for goal_policy in ada_handler.robot_policy.assist_policy.goal_assist_policies:
        for target_policy in goal_policy.target_assist_policies:
            target_policy.set_constants(huber_translation_linear_multiplier=1.55, huber_translation_delta_switch=0.11, huber_translation_constant_add=0.2, huber_rotation_linear_multiplier=0.20, huber_rotation_delta_switch=np.pi/72., huber_rotation_constant_add=0.3, huber_rotation_multiplier=0.20, robot_translation_cost_multiplier=14.0, robot_rotation_cost_multiplier=0.05)
    #
    print('[*] Connecting to test comms...')
    PORT_comms = 8642 #Randomly picked port for use, can be changed
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #s.connect(('172.16.0.3', PORT_comms)) #White Box
    s.connect(('192.168.1.57', PORT_comms)) #Josh laptop


    run_time =5 #how long each run should go for
    #Make first trajectory
    qdot = [0]*7
    print("Full Cycle")
    while True:
        msg = str(s.recv(2048))[2:-2]
        if len(msg)>1:
            state_str = np.array(list(map(float, (msg.split(",")) [1:8])),dtype = DoubleVar) #currently accepts a string and outputs a double converted array.
            #reset panda to transmitted position
            env.panda.reset((state_str))
            ada_handler.robot_policy.update(env.panda.state, qdot,env.panda)

            b = time.time()
            a = time.time()
            
            #While time running < 5s, run update loop of blending. 
            while(abs(a-b)<run_time):
                a = time.time()
                #update internal handler policy and retrieve action
                ada_handler.robot_policy.update(env.panda.state, qdot,env.panda)
                action = ada_handler.robot_policy.get_action()
                #step the panda environment
                env.step(joint = action,mode = 0)
            while(abs(a-b)<(run_time+2)):
                a = time.time()
            print("DONE A RUN")


            #Then do pull in next reset input and wait until time run > 7s

                

            



            


main()
