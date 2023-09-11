from interface_utils import CommClient, SimpleEnv, GUI_Interface
from model_utils import Model
import time
import numpy as np
import argparse
from utils_panda import convert_to_6d


# runs visualization suite on the client side
def main(args):
    client = CommClient(args.ip, args.port)
    env = SimpleEnv(visualize=True)
    gui = GUI_Interface()
    model = Model(args)
    VIZ_TIME_INTERVAL = 7.0  # seconds, TODO: make this an arg
    VIZ_TIME_LENGTH = 5.0
    prev_viz_time = 0.0
    # note that data is:
    # data["q"] = q
    # data["curr_pos_awrap"] = curr_pos_awrap.tolist()
    # data["trans_mode"] = trans_mode
    # data["slow_mode"] = slow_mode
    # data["curr_gripper_pos"] = curr_gripper_pos
    # to lessen communication overhead, we assume
    # trans_mode = slow_mode = curr_gripper_pos = 1
    data = {}
    data["trans_mode"] = 0.0
    data["slow_mode"] = 0.0
    data["curr_gripper_pos"] = 0.0
    while True:
        curr_time = time.time()
        if curr_time - prev_viz_time >= VIZ_TIME_INTERVAL:
            prev_viz_time = curr_time
            # visualize the robot's anticipated trajectory
            data = client.listen2comms()
            env.panda.reset(data["q"])
            start_time = time.time()
            while start_time + VIZ_TIME_LENGTH > time.time():
                alpha, action = model.get_params(data)
                # note that the action needs to be flipped, the sim is backwards
                action[0] = -action[0] # x
                action[1] = -action[1] # y
                env.panda.read_jacobian()
                J_pinv = np.linalg.pinv(env.panda.state["J"])
                qdot = J_pinv @ np.asarray(action)
                state, _, _, _ = env.step(joint=qdot, mode=0, grasp=False)
                data["q"] = state["q"][0:len(data["q"])].tolist()
                data["curr_pos_awrap"] = convert_to_6d(np.concatenate(
                    (state["ee_position"], state["ee_euler"])
                )).tolist()
                gui.insert(alpha)


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
    args = parser.parse_args()
    print(args)
    main(args)
