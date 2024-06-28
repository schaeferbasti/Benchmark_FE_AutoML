import numpy as np

import torch
import torch.nn as nn

from RL.Env.Environment import Environment
from Structures import Experience, RLDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:

    def __init__(self, env: Environment, rl_dataset: RLDataset):
        self.env = env
        self.state = torch.tensor(self.env.get_observation_space())
        self.dataset = rl_dataset

    def reset(self):
        self.state = torch.tensor(self.env.reset()[0])

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.sample()
        else:
            # state = torch.tensor([self.state])
            state = self.state
            # if device not in ['cpu']:
            #     state = state.cuda(device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=0)
            action = int(action.item())

        return action

    # @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = 'cpu') -> Experience:
        action = self.get_action(net, epsilon, device)

        actionResult = self.env.action(action)

        reward, done, new_state = actionResult
        new_state = torch.tensor(new_state, device=device)

        exp = Experience(self.state, action, reward, done, new_state)
        self.dataset.add_episode(exp)

        self.state = new_state
        if done:
            self.reset()
        return exp
