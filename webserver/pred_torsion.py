import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from torch.nn import functional as F
from torch.utils.data import Dataset
import numpy as np
import pathlib

class DistanceWindow(Dataset):
    """Extract distance window arrays"""

    def __init__(self, distance_window_path):
        self.distance_window_path = distance_window_path
        self.file_list = os.listdir(distance_window_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        arrays = np.load(os.path.join(self.distance_window_path, filename)).reshape((-1, 60))
        # mix_arrays = np.concatenate((arrays[:-1], arrays[1:]), 1)
        print(arrays.shape)
        torsions = np.load(os.path.join(self.distance_window_path, filename))

        return arrays, torsions, filename

# torch.cuda.set_device(0)


if torch.cuda.is_available():
    print('GPU available!!!')
    print('MainDevice=', device)


def swish_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)


class SplitModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_dim, output_dim):
        super().__init__()

        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self._bn1 = nn.BatchNorm1d(hidden_dim)

        self.hidden2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self._bn2 = nn.BatchNorm1d(2*hidden_dim)

        self.hidden3 = nn.Linear(2*hidden_dim, hidden_dim)
        self._bn3 = nn.BatchNorm1d(hidden_dim)



        self.extract_feature = nn.Linear(hidden_dim, feature_dim)
        self._bn4 = nn.BatchNorm1d(feature_dim)

        self.lstm = nn.LSTM(feature_dim, hidden_dim, bidirectional=True)
        self._bn5 = nn.BatchNorm1d(2 * hidden_dim)

        # self.sub_net1 = nn.Linear(2 * hidden_dim, hidden_dim)
        # self._bn_s1 = nn.BatchNorm1d(hidden_dim)
        # self.output1 = nn.Linear(hidden_dim, output_dim)
        #
        # self.sub_net2 = nn.Linear(2 * hidden_dim, hidden_dim)
        # self._bn_s2 = nn.BatchNorm1d(hidden_dim)
        # self.output2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, arrays):
        hidden1 = swish_fn(self._bn1(self.hidden1(arrays)))
        hidden2 = swish_fn(self._bn2(self.hidden2(hidden1)))
        hidden3 = swish_fn(self._bn3(self.hidden3(hidden2)))
        features = swish_fn(self._bn4(self.extract_feature(hidden3)))

        hidden, _ = self.lstm(features.view(len(features), 1, -1))
        output = swish_fn(self._bn5(hidden.squeeze(1)))

        # sub_hidden1 = swish_fn(self._bn_s1(self.sub_net1(hidden)))
        # # sub_hidden1 = F.dropout(sub_hidden1, p=0.5, training=self.training)
        # output1 = self.output1(sub_hidden1)
        #
        # sub_hidden2 = swish_fn(self._bn_s2(self.sub_net2(hidden)))
        # # sub_hidden2 = F.dropout(sub_hidden2, p=0.5, training=self.training)
        # output2 = self.output1(sub_hidden2)
        #
        # output = torch.cat([output1, output2], 1)
        return output



