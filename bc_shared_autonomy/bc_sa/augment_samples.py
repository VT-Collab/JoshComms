import numpy as np
from models import ReplayMemory, ReplayMemory_t


def add_samples_from_states(memory: ReplayMemory | ReplayMemory_t, noise=1e-3, n_gen=5):
    """
    Generate (s, a) samples by adding noise to their states and calculating
    the predicted action. 

    Param         type       {what}
    -------------------------------
    noise         float      stdev of gaussian noise to apply
    n_gen         int        number of (s, a) pairs to generate for each datapoint
    """
    raise NotImplementedError

    return


def add_samples_from_actions(memory: ReplayMemory | ReplayMemory_t, s_delta=5e-4, a_factor=1.0):
    """
    Generate and remove (s, a) samples by analyzing their actions.
    This is done in the following ways:
    1. If states are close to each other such that $s_delta < action^t$
       (i.e. the action is very small), then the state is removed
       from the dataset.
    2. Additional states are added by multiplying actions by a factor.

    Param         type       {what}
    -------------------------------
    s_delta       float      threshold to check for actions
    a_factor      float      multiplier for actions. If this is <=1, then it is ignored
    """
    raise NotImplementedError
    return
