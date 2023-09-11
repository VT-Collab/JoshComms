import numpy as np
import torch
import torch.nn.functional as F

from sari.train_classifier import Net
from sari.train_cae import CAE

# import rospy

GCL_FILENAMES = [
    "intent0/sac_gcl_FINAL_intent0.pt",
    "intent1/sac_gcl_FINAL_intent1.pt",
    "intent2/sac_gcl_FINAL_intent2.pt",
]


class Model(object):
    def __init__(self, args):
        self.args = args
        # tasks = ["place", "pour", "stir"]
        tasks = ["all"]
        cae_name = "sari/models/cae_" + "_".join(tasks[: self.args.n_intents])
        class_name = "sari/models/class_" + "_".join(tasks[: self.args.n_intents])
        print(cae_name)
        print(class_name)
        self.model = SARI(classifier_name=class_name, cae_name=cae_name)
        # rospy.loginfo("Loaded SARI model with {} intents".format(self.args.n_intents))

    def get_params(self, data):
        # classifier input = q + curr_pos_awrap
        d = data["q"] + data["curr_pos_awrap"]
        alpha = self.model.classify(d)
        # encoder input q + curr_pos_awrap + curr_gripper_pos + trans_mode + slow_mode
        d = (
            data["q"]
            + data["curr_pos_awrap"]
            + [
                data["curr_gripper_pos"],
                float(data["trans_mode"]),
                float(data["slow_mode"]),
            ]
        )
        z = self.model.encoder(d)
        a_robot = self.model.decoder(z, d)
        alpha = min(alpha, 0.6)
        return [alpha, a_robot]


class SARI(object):
    def __init__(self, classifier_name, cae_name):
        self.class_net = Net()
        self.cae_net = CAE()

        model_dict = torch.load(classifier_name, map_location="cpu")
        self.class_net.load_state_dict(model_dict)

        model_dict = torch.load(cae_name, map_location="cpu")
        self.cae_net.load_state_dict(model_dict)

        self.class_net.eval
        self.cae_net.eval

    def classify(self, c):
        labels = self.class_net.classify(torch.FloatTensor(c))
        confidence = F.softmax(labels, dim=0)
        return confidence.data[0].numpy()
        # return labels.detach().numpy()

    def encoder(self, c):
        z_mean_tensor = self.cae_net.encoder(torch.FloatTensor(c))
        return z_mean_tensor.tolist()

    def decoder(self, z, s):
        z_tensor = torch.FloatTensor(z + s)
        a_predicted = self.cae_net.decoder(z_tensor)
        return a_predicted.data.numpy()

