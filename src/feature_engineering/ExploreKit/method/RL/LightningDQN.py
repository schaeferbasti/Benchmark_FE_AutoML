from typing import Tuple

import gym as gym

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from Agent import Agent
from DQNNets import DuelingDQNNet
from Structures import RLDataset

class LightningDQN(LightningModule):

    def __init__(self, env: str = "CartPole-v1", lr: float=3e-4, gamma: float = 1.00, epsilon: float = 1.0, epsilon_decay_rate: float = 0.9999, sync_rate: int = 25):
        super().__init__()
        self.save_hyperparameters()
        self.env = gym.make(env)
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.net = DuelingDQNNet(obs_size, n_actions, 42)
        self.target_net = DuelingDQNNet(obs_size, n_actions, 42)

        self.dataset = RLDataset()
        self.agent = Agent(self.env, self.dataset)

        self.total_reward = 0
        self.episode_reward = 0

        # self.agent.play_step(self.net)
        self.populate()
    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.
        """
        for _ in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)
    def forward(self, x: Tensor) -> Tensor:
        '''Pass in a state x through the network and gets
        the q_values of each action as an output'''
        output = self.net(x.float())
        return output

    #def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        state, action, reward, done, next_state = batch

        state_action_value = self.net(state).gather(1, action.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_value = self.target_net(next_state).max(1)[0]
            next_state_value[done] = 0
            next_state_value = next_state_value.detach()

        expected_state_action_values = next_state_value * self.hparams.gamma + reward.float()
        return nn.MSELoss()(state_action_value, expected_state_action_values)

    def training_step(self, batch, nb_batch):
        device = self.get_device(batch)
        epsilon = self.get_epsilon()

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward
        self.log("episode_reward", self.episode_reward)

        # calc train loss
        loss = self.dqn_mse_loss(batch)

        if done:
            print(self.episode_reward)
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict(
            {
                'reward': reward,
                'train_loss': loss
            })
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(dataset=self.dataset, batch_size=16)
        return dataloader

    def get_epsilon(self) -> float:
        self.epsilon = max(0.1, self.epsilon * self.epsilon_decay_rate)
        return self.epsilon
        # if self.global_step > frames:
        #     return end
        # return start - (self.global_step / frames) * (start - end)
    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"
