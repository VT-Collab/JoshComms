import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import sys
import argparse
import glob
from utils_panda import convert_to_6d
from panda_env2 import Panda
#from geometry_msgs.msg import Twist

np.set_printoptions(precision=2, suppress=True)

device = "cpu"

# collect dataset
class MotionData(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        snippet = torch.FloatTensor(item[0]).to(device)
        state = torch.FloatTensor(item[1]).to(device)
        action = torch.FloatTensor(item[2]).to(device)
        return (snippet, state, action)


# conditional autoencoder
class CAE(nn.Module):

    def __init__(self):
        super(CAE, self).__init__()

        self.loss_func = nn.MSELoss()

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(19, 30),
            nn.Tanh(),
            nn.Linear(30, 40),
            nn.Tanh(),
            nn.Linear(40, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 40)
        )

        # Policy
        self.dec = nn.Sequential(
            nn.Linear(59, 40),
            nn.Tanh(),
            nn.Linear(40, 30),
            nn.Tanh(),
            nn.Linear(30, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 6)
        )

    def encoder(self, x):
        return self.enc(x)

    def decoder(self, z_with_state):
        return self.dec(z_with_state)

    def forward(self, x):
        history, state, action = x
        z =  self.encoder(history)
        z_with_state = torch.cat((z, state), 1)
        action_decoded = self.decoder(z_with_state)

        loss = self.loss(action, action_decoded)
        return loss

    def loss(self, action_decoded, action_target):
        # return self.loss_func(action_decoded, action_target)
        return self.loss_func(action_decoded[:,:3], action_target[:,:3]) + .9 * self.loss_func(action_decoded[:,3:], action_target[:,3:])

def train_cae(args):
    panda = Panda()

    parent_folder = 'demos'
    folders = ["demo_fork", "demo_cups", "demo_mug", "demo_shakers"]
    folders = [folders[args.folder_to_use]]

    data_folder = "data"
    model_folder = "models"
    savename = 'cae_' + "_".join(folders)
    lookahead = args.lookahead#5
    noiselevel = args.noiselevel#0.0005
    noisesamples = args.noisesamples#5
    dataset = []
    demos = []
    for folder in folders:
        demos += glob.glob(parent_folder + "/" + folder + "/**.pkl")
    print(demos)
    # demos = [parent_folder + "/" + folder + "/" +folder+f"_{i}"+ ".pkl" for i in range(10)]
    
    inverse_fails = 0
    for filename in demos:
        demo = pickle.load(open(filename, "rb"))
        n_states = len(demo)

        for idx in range(n_states-lookahead):

            home_pos = np.asarray(demo[idx]["start_pos"])
            home_q = np.asarray(demo[idx]["start_q"])
            home_gripper_pos = [demo[idx]["start_gripper_pos"]]
            
            curr_pos = np.asarray(demo[idx]["curr_pos"])
            curr_q = np.asarray(demo[idx]["curr_q"])
            curr_gripper_pos = [demo[idx]["curr_gripper_pos"]]
            curr_trans_mode = [float(demo[idx]["trans_mode"])]
            curr_slow_mode = [float(demo[idx]["slow_mode"])]

            next_pos = np.asarray(demo[idx+lookahead]["curr_pos"])
            next_q = np.asarray(demo[idx+lookahead]["curr_q"])
            next_gripper_pos = [demo[idx+lookahead]["curr_gripper_pos"]]
            for _ in range(noisesamples):
                noise_pos = curr_pos.copy() + np.random.normal(0, noiselevel, len(curr_pos))
                
                noise_q = np.array(panda.pose2joint(noise_pos))[0:len(curr_q)]

                if None in noise_q:
                    inverse_fails += 1
                    continue

                # Angle wrapping is a bitch
                noise_pos_awrap = convert_to_6d(noise_pos)
                # next_pos_awrap = convert_to_6d(next_pos)

                action = next_pos - noise_pos


                history = noise_q.tolist() + noise_pos_awrap.tolist() + curr_gripper_pos \
                            + curr_trans_mode + curr_slow_mode
                state = noise_q.tolist() + noise_pos_awrap.tolist() + curr_gripper_pos \
                            + curr_trans_mode + curr_slow_mode
                dataset.append((history, state, action.tolist()))

    pickle.dump(dataset, open(data_folder + "/" + savename, "wb"))
    

    model = CAE().to(device)
    train_data = MotionData(dataset)

    # EPOCH = 100
    EPOCH = 300
    # BATCH_SIZE_TRAIN = 2#int(train_data.__len__() / 10.)
    BATCH_SIZE_TRAIN = 2#int(train_data.__len__() / 10.)
    # LR = 0.0001
    LR = 0.0001
    LR_STEP_SIZE = 400
    LR_GAMMA = 0.15
    
    
    #print((train_data.__len__()))
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for epoch in range(EPOCH):
        for batch, x in enumerate(train_set):
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print("epoch: {} loss: {}".format(epoch, loss.item()))
        torch.save(model.state_dict(), model_folder + "/" + savename)

def main():
    #rospy.init_node("train_cae")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, help="number of tasks to use", default=1)
    parser.add_argument("--folder-to-use", type=int, help="which folder should \
                        we use for one by one training", default=0)
    parser.add_argument("--lookahead", type=int, help="lookahead to compute robot action", default=15)
    parser.add_argument("--noisesamples", type=int, help="num of noise samples", default=5)
    parser.add_argument("--noiselevel", type=float, help="variance for noise", default=.0001)
    args = parser.parse_args()
    train_cae(args)


if __name__ == "__main__":
    main()
    # except rospy.ROSInterruptException:
    #     pass
