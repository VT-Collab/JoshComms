from argparse import ArgumentParser
import numpy as np

from env.env2 import SimpleEnv
from bc_sa.models import RobotPolicy


def run_virtual(args):
    model = RobotPolicy()
    env = SimpleEnv(visualize=True)
    num_joints = 7
    while True:
        state = env.panda.state
        q = state["q"][0:num_joints]
        action = model(q)
        state, _, _, _ = env.step(joint=action, mode=0, grasp=False)


def run_real(args):
    raise NotImplementedError


def main(args):
    if args.virtual:
        return run_virtual(args)
    else:
        return run_real(args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--virtual", action="store_true", help="pass this flag to use a virtual panda"
    )
    parser.add_argument(
        "--model-filename", type=str, default="./bc_sa/models/bc_sa", help="pass this flag to use a virtual panda"
    )
    main(parser.parse_args())
