
#!/usr/bin/env python

#Sets up the robot, environment, and goals to use shared autonomy code for grasping objects

from AssistanceHandler import *

import numpy as np
from Goal import Goal
from Utils import *

from tf import *
#import adapy

#import prpy


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Ada Assistance Policy')
    parser.add_argument('--x', type=float, default=0.4)
    parser.add_argument('--y', type=float, default=0.3)
    parser.add_argument('--z', type=float, default=0.1)

    args = parser.parse_args()
#   goal_distribution = np.array([0.333, 0.333, 0.333])
    print('[*] Connecting to low-level controller...')
    #self.panda = panda()

    PORT_robot = 8080
                
    conn = connect2robot(PORT_robot)
    robot_state = readState(conn)


    # start_state = self.robot_state
    # start_pos,start_trans = joint2pose(self.robot_state['q'])


    action_scale = 0.1

    start_time = time.time()
    end_time=time.time()

    
    goal_position = [args.x, args.y,args.z] 
    #goal_quat = args.quat
    while True:
        
        xdot = [0]*6
        robot_dof_values = 7
        #get pose of min value target for user's goal
        robot_state = readState(conn)
        ee_pos,ee_trans = joint2pose(robot_state['q'])  

        xdot = [0]*6
        robot_dof_values = 7
        #direct_teleop_action = xdot2qdot(xdot, self.env.panda.state) #qdot
        xcurr = ee_pos
        
        poschange = goal_position - xcurr
        #qcurr =  robot_state['ee_quaternion']
        #   quat_dot = goal_quat - qcurr
        xdot = np.append(poschange,[0,0,0])
        print(ee_pos)
        action = xdot2qdot(xdot*.5,robot_state)
        
        send2robot(conn, action)
        #self.env.step(joint = action,mode = 0)
        end_time=time.time()
        if end_time-start_time > 30.0:
            print("DONE DONE DONE")
            break
        if (np.linalg.norm(poschange) < .005):
            print("AT GOAL DONE")
            break
