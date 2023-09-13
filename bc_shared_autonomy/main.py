import numpy as np
from argparse import ArgumentParser
import pickle
from interfaces.interface_utils import CommServer
from bc_sa.models import RobotPolicy
from utils_panda import JoystickControl, TrajectoryClient


def main(args):
    model = RobotPolicy()
    mover = TrajectoryClient()
    joystick = JoystickControl()

    user_data = []

    while True:

        raise NotImplementedError # TODO handle logic



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--interface",
        type=str,
        choices=["none", "rumble", "visual", "both"],
        help="interface to use for communication",
    )
    main(parser.parse_args())
