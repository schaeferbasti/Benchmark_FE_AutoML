from typing import List, Dict

import numpy as np
import torch.cuda
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence

from DQNNets import DuelingDQNNet, DQNNet
from Agent import Agent
from RL.Env.Environment import Environment
from Structures import RLDataset, Experience, Memory


class DQN():
    def __init__(self, env: Environment, lr: float=3e-4, gamma: float = 1.0, epsilon: float = 1.0,
                 epsilon_decay_rate: float = 0.9999, sync_rate: int = 25, mem_size: int = 2000, batch_size: int = 32):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.lr = lr
        self.sync_rate = sync_rate
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        obs_size = self.env.get_observation_space()
        n_actions = self.env.get_action_space()
        self.net = DQNNet(obs_size, n_actions, 42).to(self.device)
        self.target_net = DQNNet(obs_size, n_actions, 42).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())

        self.optimizer = Adam(self.net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # self.dataset = RLDataset()
        self.memory = Memory(self.mem_size)
        self.agent = Agent(self.env, self.memory)

        self.rewards: List[int] = []

    def train(self):
        self.fill_memory()

        global_step = 0
        loss_stats = []
        reward_stats = []
        actions_stats = dict.fromkeys(range(self.env.get_action_space()), 0)

        # Start Algorithm
        for episode in range(3000):
            episode_reward = 0
            steps = 0
            episode_loss = []
            local_memory = []
            current_state = self.env.reset()
            hidden_state, cell_state = self.net.init_hidden_states(bsize=1)
            # step through environment with agent
            while True:
                # choose action logic
                epsilon = self.get_epsilon()
                if np.random.rand(1) < epsilon:
                    # torch_x = torch.from_numpy(prev_state).float().to(self.device)
                    # torch_x = torch_x.view(1, 1, -1)
                    # torch_x, _ = self._pad_seqequence(torch_x)
                    # model_out = self.net.forward(torch_x, bsize=1,  hidden_state=hidden_state,
                    #                                cell_state=cell_state)
                    action = np.random.randint(0, self.env.get_action_space())
                    # hidden_state = model_out[1][0]
                    # cell_state = model_out[1][1]
                else:
                    torch_x = torch.from_numpy(current_state).float().to(self.device)
                    torch_x = torch_x.view(1, 1, -1)
                    torch_x, _ = self._pad_seqequence(torch_x)
                    out, _ = self.net.forward(torch_x, bsize=1, hidden_state=hidden_state,
                                                   cell_state=cell_state)
                    action = torch.argmax(out[0]).cpu().numpy()

                    actions_stats[action] += 1

                # Execute the game
                reward, done, next_state = self.env.action(action)
                episode_reward += reward

                # Save to replay buffer
                local_memory.append(Experience(current_state, action, reward, done, next_state))

                # CRUCIAL step easy to overlook
                current_state = next_state

                # update target network
                if (global_step % self.sync_rate) == 0:
                    self.target_net.load_state_dict(self.net.state_dict())

                # Training
                hidden_batch, cell_batch = self.net.init_hidden_states(bsize=self.batch_size)
                batch = self.memory.get_batch(bsize=self.batch_size)

                current_states = []
                acts = []
                rewards = []
                next_states = []
                dones = []

                for b in batch:
                    cs, ac, rw, ns, dones = [], [], [], [], []
                    for element in b:
                        cs.append(element[0])
                        # ac.append(element[1])
                        # rw.append(element[2])
                        ns.append(element[3])
                    current_states.append(torch.Tensor(cs))
                    # acts.append(torch.Tensor(ac))
                    # rewards.append(torch.Tensor(rw))
                    acts.append(b[-1][1])
                    rewards.append(b[-1][2])
                    dones.append(b[-1][4])
                    next_states.append(torch.Tensor(ns))

                torch_acts = torch.LongTensor(acts).to(self.device)
                torch_rewards = torch.FloatTensor(rewards).to(self.device)
                torch_dones = torch.IntTensor(dones).to(self.device)

                torch_current_states, _ = self._pad_seqequence(current_states)
                torch_current_states = torch_current_states.to(self.device)

                torch_next_states, _ = self._pad_seqequence(next_states)
                torch_next_states = torch_next_states.to(self.device)

                with torch.no_grad():
                    Q_next, _ = self.target_net.forward(torch_next_states, bsize=self.batch_size,
                                                 hidden_state=hidden_batch, cell_state=cell_batch)
                    Q_next_max, __ = Q_next.detach().max(dim=1)
                    target_values = self.gamma * Q_next_max
                target_values[torch_dones] = 0
                target_values += torch_rewards

                Q_s, _ = self.net.forward(torch_current_states, bsize=self.batch_size,
                                            hidden_state=hidden_batch, cell_state=cell_batch)
                Q_s_a = Q_s.gather(dim=1, index=torch_acts.unsqueeze(dim=1)).squeeze(dim=1)

                loss = self.criterion(Q_s_a, target_values)
                #  save performance measure
                episode_loss.append(loss.item())

                # optimize model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if done:
                    self.rewards.append(episode_reward)
                    break

                global_step += 1
                steps += 1

            self.update_stats(loss.item(), episode_reward, episode)
            reward_stats.append(episode_reward)
            loss_stats.append(np.mean(episode_loss))
            self.memory.add_episode(local_memory)

        print(actions_stats)
        self.plot({'episode_reward': reward_stats, 'episode_loss': loss_stats})

    def fill_memory(self):
        for i in range(0, self.mem_size):
            prev_state = self.env.reset()
            local_memory = []
            done = False

            while done != True:
                action = np.random.randint(0, self.env.get_action_space())
                reward, done, next_state = self.env.action(action)

                local_memory.append((prev_state, action, reward, next_state, done))

                prev_state = next_state

            self.memory.add_episode(local_memory)

    def _pad_seqequence(self, batch) -> (PackedSequence, List[int]):
        x_lens = [len(x) for x in batch]
        xx_pad = pad_sequence(batch, batch_first=True, padding_value=0)
        x_packed = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False)
        return x_packed, x_lens

    def dqn_mse_loss(self, exp: Experience) -> nn.MSELoss:
        state, action, reward, done, next_state = exp

        state_action_value = self.net(state)[action] #.gather(1, action).squeeze(-1)

        with torch.no_grad():
            next_state_value = self.target_net(next_state).max(0)[0]
            next_state_value[done] = 0
            next_state_value = next_state_value.detach()

        expected_state_action_value = next_state_value * self.gamma + reward
        return nn.MSELoss()(state_action_value, expected_state_action_value)

    def get_epsilon(self) -> float:
        self.epsilon = max(0.1, self.epsilon * self.epsilon_decay_rate)
        return self.epsilon

    def update_stats(self, loss_value, reward, episode):
        print(f'{loss_value=}, {reward=}, {episode=}, {self.epsilon}', end='\r')

    def plot(self, plots: Dict[str, list]):
        plt.figure(1)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episodes')
        for k,v in plots.items():
            plt.plot(list(range(len(v))), v, label=k)

        plt.show()

