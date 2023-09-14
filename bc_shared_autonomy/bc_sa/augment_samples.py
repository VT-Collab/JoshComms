import numpy as np
from models import ReplayMemory


def add_samples_from_states(
    memory: ReplayMemory, noise=1e-3, n_gen=50, ignore_gripper=True
):
    """
    Generate (s, a) samples by adding noise to their states and calculating
    the predicted action.

    Param         type       {what}
    -------------------------------
    noise         float      stdev of gaussian noise to apply
    n_gen         int        number of (s, a) pairs to generate for each datapoint
    """
    tmp_memory = ReplayMemory(capacity=(n_gen * (len(memory) - 1)))
    max_idx = len(memory) - 1
    for idx, datapoint in enumerate(memory.buffer):
        if idx == max_idx:
            break
        next_state = memory.buffer[idx + 1][0]
        for _ in range(n_gen):
            noise_state = np.copy(datapoint[0])
            noise_state += np.random.normal(0, noise, size=noise_state.size)
            noise_action = next_state - noise_state
            if ignore_gripper:
                noise_action = noise_action[0:-1]
            tmp_memory.push(noise_state, noise_action)
    return tmp_memory


def add_samples_from_actions(memory: ReplayMemory, s_delta=5e-3, a_factor=1.0):
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
    # this implements 1. 
    tmp_memory = ReplayMemory(capacity=memory.capacity)
    tmp_memory_capacity = 0
    for _, val in enumerate(memory.buffer):
        if np.linalg.norm(val[1]) < s_delta:
            continue
        tmp_memory.push(val[0], val[1])
        tmp_memory_capacity += 1
    # this implements 2. 
    # TODO
    final_memory = ReplayMemory(capacity=tmp_memory_capacity)
    final_memory.buffer = tmp_memory.buffer[0:tmp_memory_capacity]
    final_memory.position = tmp_memory.position
    final_memory.size = tmp_memory.size
    return final_memory

def add_samples_from_lookahead(memory: ReplayMemory, s_delta=1e-2, lookahead=5, n_runs=1):
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
    # this implements 1. 
    tmp_memory = ReplayMemory(capacity=memory.capacity)
    tmp_memory_capacity = 0
    for r in range(n_runs):
        idx = int(r * len(memory) / n_runs)
        next_idx = int((r + 1) * len(memory) / n_runs)
        for i in range(idx, next_idx - lookahead):
            state = memory.buffer[i][0]
            next_state = memory.buffer[i + lookahead][0]
            action = (next_state - state)[0:-1] # ignore gripper
            if np.linalg.norm(action) < s_delta:
                continue
            tmp_memory.push(state, action)
            tmp_memory_capacity += 1
    # this implements 2. 
    # TODO
    final_memory = ReplayMemory(capacity=tmp_memory_capacity)
    final_memory.buffer = tmp_memory.buffer[0:tmp_memory_capacity]
    final_memory.position = tmp_memory.position
    final_memory.size = tmp_memory.size
    return final_memory
