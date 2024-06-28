
from itertools import pairwise
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence

dims = [100, 10]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_size = 128

class DQNNet(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super().__init__()

        self.input_size = state_size
        self.out_size = action_size

        self.seed = torch.manual_seed(seed)
        self.seq = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=hidden_size),
            nn.LeakyReLU(0.3),
            nn.Linear(in_features=hidden_size, out_features=64),
            nn.LeakyReLU(0.3),
            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(0.3),
            nn.Linear(in_features=32, out_features=self.out_size),
        )
    def forward(self,  x: PackedSequence, bsize, hidden_state, cell_state):
        out, input_sizes = pad_packed_sequence(x, batch_first=True)
        out = out[torch.arange(out.shape[0]), input_sizes - 1]

        out = self.seq(out)
        return out, 42

class DuelingDQNNet(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DuelingDQNNet, self).__init__()

        self.input_size = state_size
        self.out_size = action_size

        self.seed = torch.manual_seed(seed)
        self.lstm_layer = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.adv = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=hidden_size),
            nn.LeakyReLU(0.3),
            nn.Linear(in_features=hidden_size, out_features=64),
            nn.LeakyReLU(0.3),
            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(0.3),
            nn.Linear(in_features=32, out_features=self.out_size),
        )
        self.val = nn.Sequential(
            nn.LeakyReLU(0.3),
            nn.Linear(in_features=hidden_size, out_features=64),
            nn.LeakyReLU(0.3),
            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(0.3),
            nn.Linear(in_features=32, out_features=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x: PackedSequence, bsize, hidden_state, cell_state):
        # x = x.view(bsize, time_step, 512)
        # x, lengths = self.pad_seqequence(x)

        # out, (h_n, c_n) = self.lstm_layer(x, (hidden_state, cell_state))
        # out, input_sizes = pad_packed_sequence(out, batch_first=True)
        # out = out[torch.arange(out.shape[0]), input_sizes-1]
        out, input_sizes = pad_packed_sequence(x, batch_first=True)
        out = out[torch.arange(out.shape[0]), input_sizes-1]

        adv_out = self.adv(out)
        val_out = self.val(out)

        qout = val_out.expand(bsize, self.out_size) + (
                    adv_out - adv_out.mean(dim=1).unsqueeze(dim=1).expand(bsize, self.out_size))

        return qout, 42  #(h_n, c_n)

    def init_hidden_states(self, bsize):
        h = torch.zeros(1, bsize, hidden_size).float().to(device)
        c = torch.zeros(1, bsize, hidden_size).float().to(device)

        return h, c

    @staticmethod
    def pad_seqequence(batch):
        x_lens = [len(x) for x in batch]
        xx_pad = pad_sequence(batch, batch_first=True, padding_value=0)
        x_packed = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False)
        return x_packed, x_lens

    # def _get_last_seq_items(self, packed): # mine
    #     sum_batch_sizes = torch.cat((
    #         torch.zeros(2, dtype=torch.int64),
    #         torch.cumsum(packed.batch_sizes, 0)
    #     ))
    #     sorted_lengths = lengths[packed.sorted_indices]
    #     last_seq_idxs = sum_batch_sizes[sorted_lengths] + torch.arange(lengths.size(0))
    #     last_seq_items = packed.data[last_seq_idxs]
    #     last_seq_items = last_seq_items[packed.unsorted_indices]
    #     return last_seq_items

