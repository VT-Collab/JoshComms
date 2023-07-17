# Standard imports
import rospy
import sys, time, pickle
import numpy as np
import argparse

#from utils import TrajectoryClient, JoystickControl
from utils_panda import *

"""
TODO
- instantiate final rotation(R, P, P_hat) outside the loop
"""


def record_demo(args):
    usernumber = sys.argv[1]
    filename = sys.argv[2]
    savename = "user/user" + str(usernumber) + "/demos/" + str(filename) + ".pkl"
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




    #mover = TrajectoryClient()

    #rospy.loginfo("Initialized, Moving Home")
    go2home(conn)
    
    #rospy.loginfo("Reached Home, waiting for input")
    
    # start_gripper_pos = None
    # while start_gripper_pos is None:
    #     start_gripper_pos = mover.robotiq_joint_state

    
    print('[*] Main loop...')
    
    state = readState(conn)
    print('[*] Waiting for start...')
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

    while True:

        state = readState(conn)
        q = state["q"].tolist()
        pose = joint2pose(state["q"])

        #u, A_pressed, stop, start = interface.input() #[dx, dy, dz], A_pressed, B_pressed, START_pressed
        curr_pos,curr_pose = joint2pose(q)
        #curr_gripper_pos = robotiq_joint_state

        axes, gripper, mode, slow, start = joystick.getInput()
        while np.sum(np.abs(axes)) < 1e-3 and not run_start:
            #axes, gripper, mode, slow, start = joystick.getInput()
            start_time = time.time()
            # assist_time = time.time()
        
        if start and not record:
            record = True
            start_time = time.time()
            print('[*] Recording the demonstration...')

        curr_time = time.time()
        if record and curr_time - start_time >= steptime:
            demonstration.append(start_q.tolist() + q)
            start_time = curr_time
        
        # actuate gripper
        gripper_ac = 0
        xdot_h = np.zeros(6)
        curr_time = time.time()

        if record:
            if gripper and gripper_ac:
                gripper_ac = 0
                send2gripper(conn_gripper, "c")
                time.sleep(0.5)
            elif gripper and not (gripper_ac):
                gripper_ac = 1
                send2gripper(conn_gripper, "c")
                time.sleep(0.5)
                # while gripper:
                #     axes, gripper, mode, slow, start = joystick.getInput()
        
            # switch between translation and rotation
            if mode:
                trans_mode = not trans_mode
               # rospy.loginfo("Translation Mode: {}".format(trans_mode))
                # while mode:
                #     axes, gripper, mode, slow, start = joystick.getInput()
            
            # Toggle speed of robot
            if slow:
                slow_mode = not slow_mode
                #rospy.loginfo("Slow Mode: {}".format(trans_mode))
                # while slow:
                #     axes, gripper, mode, slow, start = joystick.getInput()
            
            if slow_mode:
                scaling_trans = 0.1
                scaling_rot = 0.2
            else:
                scaling_trans = 0.2
                scaling_rot = 0.4

            # Get input velocities for robot            
            if trans_mode: 
                xdot_h[:3] = scaling_trans * np.asarray(axes)

            elif not trans_mode:
                # change coord frame from robotiq gripper to tool flange
                R = np.mat([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
                P = np.array([0, 0, -0.10])
                P_hat = np.mat([[0, -P[2], P[1]],
                                [P[2], 0, -P[0]],
                                [-P[1], P[0], 0]])
                
                axes = np.array(axes)[np.newaxis]
                trans_vel = scaling_rot * P_hat * R * axes.T
                rot_vel = scaling_rot * R * axes.T
                xdot_h[:3] = trans_vel.T[:]
                xdot_h[3:] = rot_vel.T[:]

            # Save waypoint
            if curr_time - start_time >= step_time:
                # rospy.loginfo("State : {}".format(curr_pos))
                data = {}
                data["start_q"] = start_q
                data["start_pos"] = start_pos
                data["start_gripper_pos"] = 0
                data["curr_q"] = q
                data["curr_pos"] = curr_pos
                data["curr_gripper_pos"] = gripper_ac
                data["trans_mode"] = trans_mode
                data["slow_mode"] = slow_mode
                data["xdot_h"] = xdot_h.tolist()
                data["gripper_ac"] = gripper_ac
                demo.append(data)
                start_time = curr_time

        qdot = xdot2qdot(xdot_h, state)
        send2robot(conn, qdot)


def main():
    rospy.init_node("record_demo")
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
        rospy.loginfo("Saved demo as: {}".format(savename))
        rospy.loginfo("Sample Datapoint : {}".format(demo[0]))
    
    else:
        rospy.loginfo("Demo not saved.")

if __name__ == "__main__":
        try:
            main()
        except rospy.ROSInterruptException:
            pass