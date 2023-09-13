from typing import assert_type
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import pickle
import glob

from models import RobotPolicy, ReplayMemory


def train_robot_policy(
    model: RobotPolicy, memory: ReplayMemory, optim: Adam, epochs=100, batch_size=1000
):
    """
    Training the robot policy via simple behavior cloning. There are better
    ways to do this, but lets start with the basics.

    Param        Type
    ------------------
    model        RobotPolicy
    memory       ReplayMemory (states, actions)
    optim        Adam
    epochs       int (100)
    batch_size   int (1000)
    """
    for ep in range(epochs):
        states, actions = memory.sample(batch_size)
        state_batch = torch.FloatTensor(states)
        action_batch = torch.FloatTensor(actions)

        action_hat = model(state_batch)

        loss = nn.functional.mse_loss(action_hat, action_batch)
        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f"Epoch: {ep}, Loss: {np.round(loss.item(), 2)}")

    return


def main(args):
    memory = ReplayMemory(capacity=0)  # buffer acts as a list
    model = RobotPolicy()
    optim = Adam(model.parameters(), lr=0.001)
    files = glob.glob(args.demo_folder + "/*.pkl")
    for f in files:
        data = pickle.load(f)
        assert type(data) == ReplayMemory
        memory.buffer = np.concatenate((memory.buffer, data.buffer))
    memory.size = len(memory.buffer)
    memory.position = len(memory.buffer)
    train_robot_policy(
        model, memory, optim, epochs=args.epochs, batch_size=args.batch_size
    )
    model.save(args.model_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--demo-folder", type=str, default="demos", help="file to load demos from"
    )
    parser.add_argument(
        "--model-file", type=str, default="bc_model", help="file to save model to"
    )
    parser.add_argument("--epochs", type=int, default=100, help="epochs to train")
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="batch size to sample"
    )
    main(parser.parse_args())

