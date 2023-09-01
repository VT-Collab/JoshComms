# Standard imports
# import rospy
import os, time, pickle, argparse
import numpy as np

# Imports from current directory
# from utils import TrajectoryClient, JoystickControl, convert_to_6d
from utils_panda import JoystickControl, convert_to_6d
from utils_panda import (
    listen2robot,
    joint2pose,
    xdot2qdot,
    send2robot,
    connect2robot,
    readState,
    go2home,
)
from model_utils import Model

# from comms_client import connect2comms, send2comms
from viz import VizServer

""" TODO
- Add event logging
"""


class TrajectoryClient:
    def __init__(self, PORT=8080):
        self.conn = connect2robot(PORT)
        self.HOME = [0.8385, -0.0609, 0.2447, -1.5657, 0.0089, 1.5335, 1.8607]
        self._get_states()  # assignment unnecessary

    def _get_states(self):
        # state = listen2robot(self.conn)
        state = readState(self.conn)  # should this be used?
        self.joint_states = state['q']
        print(state)
        print('wta')
        return state

    def _joint_to_pose(self, q):
        pos, rot = joint2pose(q)
        print(pos.tolist())
        print(rot.tolist())
        print(pos.tolist() + rot.flatten().tolist())
        return pos.tolist() + rot.flatten().tolist()

    def joint2pose(self):
        self._get_states()
        x = self._joint_to_pose(self.joint_states)
        print(x)
        return x

    def robotiq_joint_state(self):
        return 0.0

    def xdot2qdot(self, xdot):
        state = self._get_states()  # this also sets self.state
        print(xdot)
        print(state)
        return xdot2qdot(xdot, state)

    def switch_controller(self, mode):
        self.mode = mode

    def send_joint(self, q, lim):
        send2robot(self.conn, qdot, limit=lim)
        return

    def actuate_gripper(self, *args):
        return

    def send(self, qdot):
        send2robot(self.conn, qdot)
        return

    def go2home(self):
        go2home(self.conn)
        return


def run_test(args):
    # mover = TrajectoryClient()
    joystick = JoystickControl()
    mover = TrajectoryClient()

    # rate = rospy.Rate(1000)

    visualize = not args.noviz
    viz_server = None
    if visualize:
        viz_server = VizServer(args.ip, args.port)
    VIZ_TIME_INTERVAL = 7.0  # seconds

    # get savenames and print savenames
    # Create model based on runtype

    model = Model(args)

    print("Initialized, Moving Home")
    mover.go2home()
    # reset_gripper(conn)
    print("Reached Home, waiting for input")

    start_pos = None
    start_q = None
    while start_pos is None:
        start_pos = mover.joint2pose()
        start_q = mover.joint_states

    start_gripper_pos = None
    while start_gripper_pos is None:
        start_gripper_pos = mover.robotiq_joint_state()

    step_time = 0.1
    start_time = time.time()
    assist_time = time.time()

    scaling_trans = 0.2
    scaling_rot = 0.4

    trans_mode = True
    slow_mode = False
    traj = []

    assist = False
    assist_start = 1.0

    run_start = False

    folder = "./user_data/user" + str(args.user)
    abs_path = os.path.abspath(folder)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)

    savename = folder + "/" + args.filename + "_" + str(args.run_num) + ".pkl"
    while True:
        # while not rospy.is_shutdown():
        q = np.asarray(mover.joint_states).tolist()
        print(q)
        curr_pos = mover.joint2pose()
        curr_gripper_pos = mover.robotiq_joint_state()

        axes, gripper, mode, slow, start = joystick.getInput()

        # Wait for human to start moving
        while np.sum(np.abs(axes)) < 1e-3 and not run_start:
            axes, gripper, mode, slow, start = joystick.getInput()
            start_time = time.time()
            assist_time = time.time()

        if not run_start:
            print("Start received")
            run_start = True

        if start:
            pickle.dump(traj, open(savename, "wb"))
            print("Collected {} datapoints and saved at {}".format(len(traj), savename))
            mover.switch_controller(mode="position")
            mover.send_joint(q, 1.0)
            return 1

        # switch between translation and rotation
        if mode:
            trans_mode = not trans_mode
            print("Translation Mode: {}".format(trans_mode))
            while mode:
                axes, gripper, mode, slow, start = joystick.getInput()

        # Toggle speed of robot
        if slow:
            slow_mode = not slow_mode
            print("Slow Mode: {}".format(trans_mode))
            while slow:
                axes, gripper, mode, slow, start = joystick.getInput()

        if slow_mode:
            scaling_trans = 0.1
            scaling_rot = 0.2
        else:
            scaling_trans = 0.2
            scaling_rot = 0.4

        xdot_h = np.zeros(6)
        if trans_mode:
            xdot_h[:3] = scaling_trans * np.asarray(axes)
        elif not trans_mode:
            R = np.mat([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            P = np.array([0, 0, -0.10])
            P_hat = np.mat([[0, -P[2], P[1]], [P[2], 0, -P[0]], [-P[1], P[0], 0]])

            axes = np.array(axes)[np.newaxis]
            trans_vel = scaling_rot * P_hat * R * axes.T
            rot_vel = scaling_rot * R * axes.T
            xdot_h[:3] = trans_vel.T[:]
            xdot_h[3:] = rot_vel.T[:]

        qdot_h = mover.xdot2qdot(xdot_h)
        qdot_h = qdot_h.tolist()[0]

        qdot_r = [0] * 6

        gripper_ac = 0
        if gripper and (mover.robotiq_joint_state() > 0):
            mover.actuate_gripper(gripper_ac, 1, 0.0)
            while gripper:
                axes, gripper, mode, slow, start = joystick.getInput()

        elif gripper and (mover.robotiq_joint_state() == 0):
            gripper_ac = 1
            mover.actuate_gripper(gripper_ac, 1, 0)
            while gripper:
                axes, gripper, mode, slow, start = joystick.getInput()

        print(curr_pos)
        curr_pos_awrap = convert_to_6d(curr_pos)

        data = {}
        data["q"] = q
        data["curr_pos_awrap"] = curr_pos_awrap.tolist()
        data["trans_mode"] = trans_mode
        data["slow_mode"] = slow_mode
        data["curr_gripper_pos"] = curr_gripper_pos
        print(data)

        alpha, a_robot = model.get_params(data)
        xdot_r = [0] * 6
        if trans_mode:
            xdot_r[:3] = a_robot[:3]
        elif not trans_mode:
            xdot_r[3:] = a_robot[3:]

        data["alpha"] = alpha
        data["a_human"] = xdot_h.tolist()
        data["a_robot"] = xdot_r  # .tolist()

        xdot_r = mover.xdot2qdot(xdot_r)
        qdot_r = 2.0 * xdot_r
        qdot_r = qdot_r.tolist()[0]

        curr_time = time.time()
        # show visualization

        if visualize and (curr_time - prev_viz_time) >= VIZ_TIME_INTERVAL:
            viz_server.send(data)
            prev_viz_time = curr_time

        if curr_time - assist_time >= assist_start and not assist:
            print("Assistance started")
            assist = True

        if assist:
            # sac method uses an interval between assistance times
            # if args.method != "ours" :

            qdot = alpha * 1.0 * np.asarray(qdot_r) + (1 - alpha) * np.asarray(qdot_h)

            # qdot = np.clip(qdot, -0.3, 0.3)
            qdot = qdot.tolist()
        else:
            qdot = qdot_h

        data["curr_time"] = curr_time
        data["assist"] = assist
        mover.send(qdot)

        if curr_time - start_time >= step_time:
            traj.append(data)
            start_time = curr_time

        rate.sleep()


def main():
    # rospy.init_node("run")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-intents",
        type=int,
        help="Number of intents for the model for the robot (default: 1)",
        default=1,
    )
    parser.add_argument(
        "--user",
        type=int,
        help="User number for data collections (default: 0)",
        default=0,
    )
    parser.add_argument(
        "--filename", type=str, help="Savename for data (default:test)", default="test"
    )
    parser.add_argument("--ip", type=str, help="IP to use", default="127.0.0.1")
    parser.add_argument("--port", type=int, help="Port to use", default=8642)
    parser.add_argument(
        "--run-num", type=int, help="run number to save data (default:0)", default=0
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["sari"],
        help="method to use (default:sari)",
        default="sari",
    )
    parser.add_argument("--noviz", action="store_true")
    args = parser.parse_args()
    print(args)
    run_test(args)


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    main()
