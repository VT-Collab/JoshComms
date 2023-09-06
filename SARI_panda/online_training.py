import sari.train_cae
import sari.train_classifier
from sari.panda_env2 import Panda
from sari.utils_panda import convert_to_6d
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import optim


# NOTE: this can be optimized further!
def train_cae_online(
    model,
    datapoint,
    panda,
    noisesamples=5,
    noiselevel=0.005,
    EPOCH=1,
    LR=0.001,
    LR_STEP_SIZE=400,
    LR_GAMMA=0.15,
):
    dataset = []
    for _ in range(noisesamples):
        curr_pos = np.asarray(datapoint["curr_pos"])
        curr_q = np.asarray(datapoint["curr_q"])
        curr_gripper_pos = [datapoint["curr_gripper_pos"]]
        curr_trans_mode = [float(datapoint["trans_mode"])]
        curr_slow_mode = [float(datapoint["slow_mode"])]
        for _ in range(noisesamples):
            noise_pos = curr_pos.copy() + np.random.normal(0, noiselevel, len(curr_pos))
            noise_q = np.array(panda.pose2joint(noise_pos))[0 : len(curr_q)]

            if None in noise_q:
                continue

            noise_pos_awrap = convert_to_6d(noise_pos)

            action = 0.0 * noise_q  # not used in training

            history = (
                noise_q.tolist()
                + noise_pos_awrap.tolist()
                + curr_gripper_pos
                + curr_trans_mode
                + curr_slow_mode
            )
            state = (
                noise_q.tolist()
                + noise_pos_awrap.tolist()
                + curr_gripper_pos
                + curr_trans_mode
                + curr_slow_mode
            )
            dataset.append((history, state, action.tolist()))
    train_data = sari.train_cae.MotionData(dataset)

    train_set = DataLoader(dataset=train_data, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA
    )

    for _ in range(EPOCH):
        for _, x in enumerate(train_set):
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return


# the classifier must be trained on a MULTITUDE of datapoints:
# it must be trained offline if online speed is desired
def train_classifier_online(
    model,
    datapoint,
    panda,
    noisesamples=5,
    noiselevel=0.005,
    EPOCH=1,
    LR=0.001,
    LR_STEP_SIZE=400,
    LR_GAMMA=0.15,
):
    raise (NotImplementedError)


# this is a demo function that trains the SARI cae and classifier off of a
# single datapoint
def train_online(model, datapoint, panda, noisesamples=5, noiselevel=0.005, epochs=1):
    train_cae_online(
        model,
        datapoint,
        panda,
        noisesamples=noisesamples,
        noiselevel=noiselevel,
        EPOCH=epochs,
    )
    # train_classifier_online(
    #     model,
    #     datapoint,
    #     panda,
    #     noisesamples=noisesamples,
    #     noiselevel=noiselevel,
    #     EPOCH=epochs,
    # )
    return
