import numpy as np
from argparse import ArgumentParser
import torch
from interfaces.interface_utils import CommServer
from bc_sa.models import RobotPolicy, GaussianRobotPolicy
from utils_panda import JoystickControl, TrajectoryClient

from env.env2 import SimpleEnv
from bc_sa.models import RobotPolicy


def run_virtual(args):
    model = GaussianRobotPolicy()
    model.load(args.model_filename)
    env = SimpleEnv(visualize=args.visualize)
    num_joints = 7
    state = env.panda.state
    while True:
        q = state["q"][0:num_joints]
        q_tensor = torch.FloatTensor(q)
        action = model(q_tensor).detach().numpy()
        print(q, action)
        state, _, _, _ = env.step(joint=action, mode=0, grasp=False)


def run_real(args):
    raise NotImplementedError
    model = GaussianRobotPolicy()
    mover = TrajectoryClient()
    joystick = JoystickControl()



    user_data = []

    while True:
        raise NotImplementedError  # TODO handle logic


def main(args):
    if args.virtual:
        return run_virtual(args)
    else:
        return run_real(args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--interface",
        type=str,
        choices=["none", "rumble", "visual", "both"],
        help="interface to use for communication",
    )
    parser.add_argument(
        "--virtual", action="store_true", help="pass this flag to use a virtual panda"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="pass this flag to use visualize a virtual panda"
    )
    parser.add_argument(
        "--model-filename",
        type=str,
        default="bc_sa/saved_models/bc_model",
        help="pass this flag to use a virtual panda",
    )
    main(parser.parse_args())
