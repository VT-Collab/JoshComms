import socket
import numpy as np
import time

from AssistanceHandler import *

from Utils import *
from Primary_AssistPolicy import *
from UserBot import *

#GUI USE
from tkinter import *

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
				

def listen2comms(PORT):
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect(('127.0.0.1', PORT))
    #s.connect(('172.16.0.3', PORT))
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
    s.bind(('127.0.0.1', PORT))
    #s.bind(('172.16.0.3', PORT))
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
    s.connect(('127.0.0.1', PORT_comms)) #White Box
    interface = Joystick()


    run_time =2 #how long each run should go for
    #Make first trajectory
    qdot = [0]*7
    #
    MaxConf = 0.00

    GUI_1 = GUI_Interface()
    GUI_1.root.geometry("+100+100")
    GUI_1.myLabel1 = Label(GUI_1.root, text = "Target", font=("Palatino Linotype", 40))
    GUI_1.myLabel1.grid(row = 0, column = 0, pady = 50, padx = 50)
    GUI_1.root.update()

    goal_distribution = ada_handler.robot_policy.goal_predictor.get_distribution()
    max_prob_goal_ind = np.argmax(goal_distribution)
    curr_goal = goals[max_prob_goal_ind]
    name = curr_goal.name
    GUI_1.textbox1.delete(0, END)
    GUI_1.textbox1.insert(0, name)
    GUI_1.root.update()
                
    print("Full Cycle")
    a = time.time() 
    while True:
        msg = str(s.recv(2048))[2:-2]
        if len(msg)>1:
            #msg=msg.replace(","," ")
            msg=msg.replace("\\n" , "" )
            #msg=msg.replace("n","")
            b = time.time() 
            #print(abs(a-b))
            state_str = np.array(list(map(float, (msg.split(",")) [1:8])),dtype = DoubleVar) #currently accepts a string and outputs a double converted array.
            log_goal_distribution= np.array(list(map(float, (msg.split(",")) [8:18])),dtype = DoubleVar)
            #reset panda to transmitted position
            env.panda.reset((state_str))
            ada_handler.robot_policy.goal_predictor.log_goal_distribution = list(log_goal_distribution)	
            ada_handler.robot_policy.update(env.panda.state, qdot)
            
            #While time running < runtime, run update loop of blending. 
            while(abs(a-b)<run_time):
                a = time.time()
                #update internal handler policy and retrieve action
                ada_handler.robot_policy.update(env.panda.state, qdot)
                action = ada_handler.robot_policy.get_blend_action_confident()*3
                env.step(joint = action,mode = 0)
            
        
            goal_distribution = ada_handler.robot_policy.goal_predictor.get_distribution()
            if ada_handler.robot_policy.blend_confidence_function_prob_diff(goal_distribution):
                GUI_1.fg = '#00ff00'
                GUI_1.textbox1 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
                GUI_1.textbox1.grid(row = 1, column = 0,  pady = 10, padx = 20)
            else:
                GUI_1.fg = '#ff0000'
                GUI_1.textbox1 = Entry(GUI_1.root, width = 8, bg = "white", fg=GUI_1.fg, borderwidth = 3, font=("Palatino Linotype", 40))
                GUI_1.textbox1.grid(row = 1, column = 0,  pady = 10, padx = 20)                
            
            max_prob_goal_ind = np.argmax(goal_distribution)
            curr_goal = goals[max_prob_goal_ind]
            name = curr_goal.name
            #print(name,goal_distribution)	
            #goal_distribution_sorted = np.sort(goal_distribution)

            #MaxConf = goal_distribution[max_prob_goal_ind]
            GUI_1.textbox1.delete(0, END)
            GUI_1.textbox1.insert(0, name)
            GUI_1.root.update()
                
            print("DONE A RUN")
    return -1

            #Then do pull in next reset input and wait until time run > 7s

                

            



            


main()
