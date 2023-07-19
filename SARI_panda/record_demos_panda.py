# Standard imports

import sys, time, pickle
import numpy as np
import argparse

#from utils import TrajectoryClient, JoystickControl
from utils_panda import *



def record_demo(args):
    usernumber = sys.argv[1]
    filename = sys.argv[2]
    savename  = args.save_loc + "/" + args.user + "/" + args.name + ".pkl"
    print('[*] Connecting to low-level controller...')
    PORT = 8080
    PORT_gripper = 8081
    conn = connect2robot(PORT)
    joystick = JoystickControl()
    #Connecting to grippper
    print('[*] Connecting to Gripper')
    conn_gripper = connect2gripper(PORT_gripper)
    print('[*] Initializing recording...')
    demonstration = []
    record = False
    translation_mode = True
    start_time = time.time()
    scaling_trans = 0.1
    scaling_rot =0.2
    steptime = 0.1
    gripper_start = 0
    
    if gripper_start == 1:
        send2gripper(conn_gripper, "c")
    else:
        send2gripper(conn_gripper, "o")

    #       go2home(conn)
    

    print('[*] Main loop...')
    
    state = readState(conn)
    
    start_q = np.asarray(state["q"])
    start_pos,start_pose = joint2pose(start_q)

    scaling_trans = 0.2
    scaling_rot = 0.4
    step_time = 0.1
    record = False
    trans_mode = True
    slow_mode = False
    demo = []
    run_start = False
    start_press = False
    temp_delay1 = 0
    temp_delay2 = 0
    temp_delay3 = 0
    temp_delay4 = 0
    gripper_ac = 0
    print('[*] Waiting for start...')
    while True:

        state = readState(conn)
        q = state["q"].tolist()
        #curr_position,curr_pose = joint2pose(state["q"])
        curr_pos = state["x"]
        
        #curr_gripper_pos = robotiq_joint_state

        axes, gripper, mode, slow, start = joystick.getInput() #tuple(z), A_pressed, B_pressed, X_pressed, stop
        #print(axes)
        if start and abs(curr_time - temp_delay4) > .35:
            temp_delay4 = time.time()
            start_press = not start_press
            print("PROOF THAT ALIENS ARE REAL .... FINALLY!")
            print(start_press)
            print(record)
        if start_press and not record:
            record = True
            start_time = time.time()
            print('[*] Recording the demonstration...')
        
        
        #curr_time = time.time()
        # if record and curr_time - start_time >= steptime:
        #     demonstration.append(start_q.tolist() + q)
        #     start_time = curr_time
        
        # actuate gripper
        
        xdot_h = [0]*6
        curr_time = time.time()

        # switch between translation and rotation
        if mode and abs(curr_time - temp_delay1) > .35:
            trans_mode = not trans_mode
            temp_delay1 = time.time()
            #print('trans')
            # rospy.loginfo("Translation Mode: {}".format(trans_mode))
            # while mode:
            #     axes, gripper, mode, slow, start = joystick.getInput()
        
        # Toggle speed of robot
        if slow and abs(curr_time - temp_delay2) > .35:
            slow_mode = not slow_mode
            temp_delay2 = time.time()
        
            
            #rospy.loginfo("Slow Mode: {}".format(trans_mode))
            # while slow:
            #     axes, gripper, mode, slow, start = joystick.getInput()
        
        if slow_mode:
            scaling_trans = 0.05
            scaling_rot = 0.1
            #print("slow")
            
        else:
            scaling_trans = 0.1
            scaling_rot = 0.2
            #print("fast")
        if gripper and abs(curr_time - temp_delay3) > .35:
#            print("Grippin")
            temp_delay3 = time.time()
            if gripper_ac == 1:
                gripper_ac = 0
                send2gripper(conn_gripper, "o")
                print("open")
            else:
                gripper_ac = 1
                send2gripper(conn_gripper, "c")
                print("close")
            time.sleep(0.5)

        # Get input velocities for robot            
        if trans_mode: 
            xdot_h[:3] = scaling_trans * np.asarray(axes)
            #print("trans")

        elif not trans_mode:
            # change coord frame from robotiq gripper to tool flange
            R = np.mat([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
            P = np.array([0, 0, -0.10])
            P_hat = np.mat([[0, -P[2], P[1]],
                            [P[2], 0, -P[0]],
                            [-P[1], P[0], 0]])
            #print("Rot")
            axes = np.array(axes)[np.newaxis]
            trans_vel = scaling_rot * P_hat * R * axes.T
            rot_vel = scaling_rot * R * axes.T
            xdot_h[:3] = np.ravel(trans_vel.T[:])
            xdot_h[3:] = np.ravel(rot_vel.T[:])

        

        # Save waypoint
        if curr_time - start_time >= step_time:
            # rospy.loginfo("State : {}".format(curr_pos))
            data = {}
            data["start_q"] = start_q
            data["start_pos"] = start_pos
            data["start_gripper_pos"] = gripper_start
            data["curr_q"] = q
            data["curr_pos"] = curr_pos
            data["curr_gripper_pos"] = gripper_ac
            data["trans_mode"] = trans_mode
            data["slow_mode"] = slow_mode
            data["xdot_h"] = xdot_h
            data["gripper_ac"] = gripper_ac
            demo.append(data)
            start_time = curr_time
        if not start_press and record:            
            pickle.dump( demo, open( savename, "wb" ) )
    
            print("[*] Done!")
            print("[*] I recorded this many datapoints: ", len(demonstration))
            print("[*] Saved file at: ", savename)
            
            return True
            

        qdot = xdot2qdot(xdot_h, state)
        send2robot(conn, qdot)


def main():
    #rospy.init_node("record_demo")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="save name")
    parser.add_argument("--save-loc", type=str, default="./user_data", help="save location for demo")
    parser.add_argument("--user", type=str, default="user0", help="user number")
    parser.add_argument("--store", action="store_true", help="use to store demo or discard")
    args = parser.parse_args()
    
    if args.store:
        assert args.name is not None, "Please enter a valid name to save the demo"

    demo = record_demo(args)

    if args.store:
        savename  = args.save_loc + "/" + args.user + "/" + args.name + ".pkl"
        pickle.dump(demo, open(savename, "wb"))
        #rospy.loginfo("Saved demo as: {}".format(savename))
        #rospy.loginfo("Sample Datapoint : {}".format(demo[0]))
    
    #else:
        #rospy.loginfo("Demo not saved.")

if __name__ == "__main__":
    main()
        #try:
            #main()
        #except rospy.ROSInterruptException:
            #pass