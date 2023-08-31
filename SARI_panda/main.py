from argparse import ArgumentParser
from run import run_test as run_sari


def main(args):
    np.set_printoptions(precision=2, suppress=True)
    rospy.init_node("run")
    rospy.loginfo(args)
    run_sari(args)


if __name__ == "__main__":
    parser = ArgumentParser("SARI Assistance Policy")
    # todo: add args
    parser.add_argument("--episodes", type=int, default=1, help="number of episodes")
    parser.add_argument(
        "--randomize-goal",
        action="store_true",
        help="pass this flag to disable visualization",
    )
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
    parser.add_argument("--noviz", type=str, action="store_true")
    args = parser.parse_args()
    main(args)

