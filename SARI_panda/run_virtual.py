import numpy as np
import time
from argparse import ArgumentParser

import pygame

from utils_panda import convert_to_6d, JoystickControl
from utils_panda import (
    joint2pose,
)
from model_utils import Model

from viz import SimpleEnv


class FakeJoystickControl:
    def __init__(self):
        pygame.display.set_mode((1, 1))

    def getInput(self):
        axes = [0.0] * 3
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    axes[2] = 1.0
                elif event.key == pygame.K_s:
                    axes[2] = -1.0
                if event.key == pygame.K_UP:
                    axes[1] = 1.0
                elif event.key == pygame.K_DOWN:
                    axes[1] = -1.0
                if event.key == pygame.K_RIGHT:
                    axes[0] = 1.0
                elif event.key == pygame.K_LEFT:
                    axes[0] = -1.0

        return axes, None, None, None, None

    def badGetInput(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                self.k_w = event.key == pygame.K_w
                self.k_s = event.key == pygame.K_w
                self.k_up = event.key == pygame.K_w
                self.k_down = event.key == pygame.K_w
                self.k_left = event.key == pygame.K_w
                self.k_right = event.key == pygame.K_w
            elif event.type == pygame.KEYUP:
                self.k_w = not (not event.key == pygame.K_w)
                self.k_s = not (not event.key == pygame.K_w)
                self.k_up = not (not event.key == pygame.K_w)
                self.k_down = not (not event.key == pygame.K_w)
                self.k_left = not (not event.key == pygame.K_w)
                self.k_right = not (not event.key == pygame.K_w)

        axes = [
            1.0 * self.k_right - 1.0 * self.k_left,
            1.0 * self.k_up - 1.0 * self.k_down,
            1.0 * self.k_w - 1.0 * self.k_s,
        ]
        return axes, None, None, None, None


def xdot2qdot(panda, xdot):
    """
    Returns qdot given xdot.

    Parameters
    -----------
    `panda` : `Panda`
        panda to read jacobian or state from
    `xdot` : `np.array`
        to convert to qdot
    Returns
    -----------
    `qdot` : `np.array`
        velocity of arm in joint space
    """
    panda.read_jacobian()
    state = panda.state
    jac = state["J"]
    J_pinv = np.linalg.pinv(jac)
    return J_pinv @ np.asarray(xdot)


def main(args):
    model = Model(args)
    env = SimpleEnv(visualize=True)
    joystick = None
    if args.nojoystick:
        joystick = FakeJoystickControl()
    else:
        joystick = JoystickControl()
    START_TIME = time.time()
    TIMEOUT_INTERVAL = 30
    trans_mode = True
    slow_mode = False
    curr_gripper_pos = 0.0
    scaling_trans = 0.2
    scaling_rot = 0.4
    NUM_JOINTS = 7
    # while time.time() - START_TIME < TIMEOUT_INTERVAL:
    while True:
        axes, _, _, _, _ = joystick.getInput()
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
        state = env.panda.state
        qdot_h = xdot2qdot(env.panda, xdot_h)[0:NUM_JOINTS]
        q = state["q"][0:NUM_JOINTS]
        xyz, rot = joint2pose(q)
        pose = xyz.tolist() + rot.flatten().tolist()
        data = {
            "q": q.tolist(),
            "curr_pos_awrap": convert_to_6d(pose).tolist(),
            "trans_mode": trans_mode,
            "slow_mode": slow_mode,
            "curr_gripper_pos": curr_gripper_pos,
        }
        alpha, action_t = model.get_params(data)
        xdot_r = [0.0] * (NUM_JOINTS - 1)
        if trans_mode:
            xdot_r[:3] = action_t[:3]
        else:
            xdot_r[3:] = action_t[3:]
        qdot_r = xdot2qdot(env.panda, xdot_r)
        # qdot_r *= 2.0
        # alpha_t = 1.0
        alpha_t = alpha
        # last joint is the gripper
        qdot_r = qdot_r.tolist()
        if np.linalg.norm(qdot_h) < 0.01:
            action = alpha_t * np.array(qdot_r) + (1 - alpha_t) * np.array(qdot_h)
        else:
            action = alpha * np.array(qdot_r) + (1 - alpha) * np.array(qdot_h)
        clipped_action = action[0:-1]
        print("alpha:", alpha)
        print("qdot_r:", np.round(qdot_r, 3))
        print("xdot_r:", np.round(xdot_r, 3))
        print("qdot_h:", np.round(qdot_h, 3))
        print("xdot_h:", np.round(xdot_h, 3))
        # state, _, _, _ = env.step(joint=action, mode=0, grasp=False)
        state, _, _, _ = env.step(joint=clipped_action, mode=0, grasp=False)


if __name__ == "__main__":
    parser = ArgumentParser()
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
    parser.add_argument("--nojoystick", action="store_true")
    parser.add_argument("--noviz", action="store_true")
    args = parser.parse_args()
    main(args)
