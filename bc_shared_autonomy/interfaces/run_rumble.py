from argparse import ArgumentParser
import time
from SARI_panda.utils_panda import Joystick
from interface_utils import CommClient

def main(args):
    client = CommClient(args.ip, args.port)
    joystick = Joystick() # is this thread-safe / blocking? TODO
    # client expects to receive a single float. Apply rumble proportional to
    # the confidence
    while True:
        confidence = client.listen2comms()
        if confidence > args.rumble_at_confidence:
            joystick.rumble(200) # rumble for 200ms






if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8641)
    parser.add_argument("--rumble-time-length", type=float, default=200.0)
    parser.add_argument("--rumble-at-confidence", type=float, default=0.5)
    main(parser.parse_args())
