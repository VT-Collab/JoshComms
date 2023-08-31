from viz import VizClient
from model_utils import Model
import time
import torch


# runs visualization suite on the client side
def main(args):
    client = VizClient(args.ip, args.port)
    VIZ_TIME_INTERVAL = 7.0  # seconds, TODO: make this an arg
    model = Model(args)
    curr_time = time.time()
    prev_viz_time = 0.0
    while True: # yuck
        if curr_time - prev_viz_time >= VIZ_TIME_INTERVAL:
            prev_viz_time = curr_time
            # visualize the robot's anticipated trajectory
            client_data_str = client()
            # note that data is:
            # data["q"] = q
            # data["curr_pos_awrap"] = curr_pos_awrap.tolist()
            # data["trans_mode"] = trans_mode
            # data["slow_mode"] = slow_mode
            # data["curr_gripper_pos"] = curr_gripper_pos
            # to lessen communication overhead, we assume 
            # trans_mode = slow_mode = curr_gripper_pos = 0




if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
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
    parser.add_argument("--port", type=int, help="Port to use", default=1234)
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
    main(parser.parse_args())
