import random
from collections import namedtuple, deque
from typing import Iterator, Tuple, List

import numpy as np
from torch.utils.data import IterableDataset


# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)

class Memory:

    def __init__(self, memsize):
        self.memsize = memsize
        self.memory = deque(maxlen=self.memsize)

    def add_episode(self, epsiode: List[Experience]):
        self.memory.append(epsiode)

    def get_batch(self, bsize) -> List[List[Experience]]:
        sampled_epsiodes = random.sample(self.memory, bsize)
        batch = []
        for episode in sampled_epsiodes:
            batch.append(episode)
        return batch

class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """
    def __init__(self) -> None:
        # self.experience: Experience = None
        self.buffer = ReplayBuffer(5 * (10 ** 4))

    def __iter__(self) -> Iterator[Tuple]:
        # state, action, reward, done, new_state = self.experience
        # yield state, action, reward, done, new_state
        states, actions, rewards, dones, new_states = self.buffer.sample(200)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]

    def set_new_experience(self, exp: Experience):
        # self.experience = exp
        self.buffer.append(exp)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.
    Args:
        capacity: size of the buffer
    """
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.
        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(next_states),
        )
