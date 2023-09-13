import numpy as np
from scipy.stats import halfnorm
import torch
import torch.nn as nn
from torch.optim import Adam


# uniformally distributes network weights at initialization
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class RobotPolicy(nn.Module):
    def __init__(
        self,
        num_joints=7,
        num_actions=7,
        n_hidden_layers=2,
        hidden_dim=256,
        activation_function=torch.relu,
    ) -> None:
        """
        num_joints: number of inputs (joints)
        num_actions: number of outputs (joints)
        n_hidden_layers: number of hidden layers to use
        hidden_dim: dimension to use for the hidden layers
        activation_function: activation function to use. defaults to relu
        """
        super(RobotPolicy, self).__init__()
        self.num_joints = num_joints
        self.num_actions = num_actions
        self.n_hidden_layers = n_hidden_layers
        self.linear_1 = nn.Linear(num_joints, hidden_dim)
        self.linear_f = nn.Linear(hidden_dim, num_actions)
        self.hidden_layers = [
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)
        ]
        self.apply(weights_init_)
        self.activation_function = activation_function

    def forward(self, state):
        """
        Map from states to actions
        """
        x = self.activation_function(self.linear_1(state))
        for i in range(self.n_hidden_layers):
            x = self.activation_function(self.hidden_layers[i](x))
        action = self.linear_f(x)
        return action

    def load(self, path: str):
        """Loads the model's state_dict from a path and applys weights"""
        self.load_state_dict(torch.load(path))
        self.eval()
        return

    def save(self, path: str):
        """Saves the model's state_dict to a path"""
        torch.save(self.state_dict, path)
        return


# ripped from LIMIT
class ReplayMemory:
    def __init__(self, capacity=1000):
        """
        Initializes the `ReplayMemory` object. If `capacity` is 0, 
        the storage becomes infinite (ie: the buffer acts as a list)

        Params           Type
        ---------------------
        capacity         int
        """
        self.capacity = int(capacity)
        self.position = 0
        self.size = 0
        self.buffer = np.zeros(self.capacity, dtype=tuple)

    def push(
            self, states: np.ndarray, actions: np.ndarray
    ) -> None:
        """
        Push a record into the buffer.

        Params           Type
        ---------------------
        TODO
        """
        if self.capacity > 0:
            self.buffer[self.position] = (states, actions)
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        else:
            self.buffer[self.position] = (states, actions)
            self.position += 1
            self.size += 1

    def sample(self, batch_size: int):
        """
        Samples records from the buffer according to a uniform distribution

        Params           Type
        ---------------------
        batch_size       int
        stdev            number
        """
        batch = np.random.choice(self.buffer[0 : self.size], batch_size)
        states, actions = map(np.stack, zip(*batch))
        return states, actions

    def weighted_sample(self, batch_size: int, stdev=10.0):
        """
        Samples records from the buffer according to a halfnorm distribution.

        Params           Type
        ---------------------
        TODO
        """
        upper_limit = self.capacity
        if self.capacity <= 0:
            upper_limit = self.size
        weights = np.array(
            halfnorm.pdf(np.arange(0, upper_limit), loc=0, scale=stdev)
        )
        weights = weights.take(np.arange(self.position - upper_limit, self.position))[
            ::-1
        ][0 : self.size]
        weights /= np.sum(weights)
        batch = np.random.choice(self.buffer[0 : self.size], batch_size, p=weights)
        states, actions = map(np.stack, zip(*batch))
        return states, actions

    def __len__(self) -> int:
        """
        Returns the number of records in the buffer.
        """
        return self.size

class ReplayMemory_t(ReplayMemory):
    """
    This is a variant of the ReplayMemory class which includes timestep and
    global indentifier data (eg, (s,a) -> (s, a, timestamp)).
    Note that the global indentifier is *unique to the instance of the object*.
    """
    def push(
            self, states: np.ndarray, actions: np.ndarray, t: float
    ) -> None:
        """
        Push a record into the buffer.

        Params           Type
        ---------------------
        TODO
        """
        if self.capacity > 0:
            self.buffer[self.position] = (states, actions, t)
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        else:
            self.buffer[self.position] = (states, actions, t)
            self.position += 1
            self.size += 1

    def sample(self, batch_size: int):
        """
        Samples records from the buffer according to a uniform distribution

        Params           Type
        ---------------------
        batch_size       int
        stdev            number
        """
        batch = np.random.choice(self.buffer[0 : self.size], batch_size)
        states, actions, t = map(np.stack, zip(*batch))
        return states, actions, t

    def weighted_sample(self, batch_size: int, stdev=10.0):
        """
        Samples records from the buffer according to a halfnorm distribution.

        Params           Type
        ---------------------
        TODO
        """
        upper_limit = self.capacity
        if self.capacity <= 0:
            upper_limit = self.size
        weights = np.array(
            halfnorm.pdf(np.arange(0, upper_limit), loc=0, scale=stdev)
        )
        weights = weights.take(np.arange(self.position - upper_limit, self.position))[
            ::-1
        ][0 : self.size]
        weights /= np.sum(weights)
        batch = np.random.choice(self.buffer[0 : self.size], batch_size, p=weights)
        states, actions, t = map(np.stack, zip(*batch))
        return states, actions, t
