import torch
import torch.nn as nn
from npy_data_loader import DistanceWindow
from torch.utils.data import DataLoader
import os
from torch.nn import functional as F
import numpy as np
import pathlib


torch.cuda.set_device(0)
device = torch.device('cuda:0')

if torch.cuda.is_available():
    print('GPU available!!!')
    print('MainDevice=', device)


train_name = 'nr40_2_Split_L1_Drop05'
save_dir = './outputs/' + train_name
val_dir = os.path.join(save_dir, 'val')

is_cross_validation = True
cross_validation_fold = 10


test_dataset = DistanceWindow(
    distance_window_path='/share/Data/processed/test_set/distance_window',
    torsion_path='/share/Data/processed/test_set/bitorsion')
test_data_loader = DataLoader(dataset=test_dataset)


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

        self.sub_net1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output1 = nn.Linear(hidden_dim, output_dim)

        self.sub_net2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, arrays):
        hidden1 = self._bn1(swish_fn(self.hidden1(arrays)))
        hidden2 = self._bn2(swish_fn(self.hidden2(hidden1)))
        hidden3 = self._bn3(swish_fn(self.hidden3(hidden2)))
        features = self._bn4(swish_fn(self.extract_feature(hidden3)))

        hidden, _ = self.lstm(features.view(len(features), 1, -1))
        hidden = self._bn5(hidden.squeeze(1))

        sub_hidden1 = self.sub_net1(hidden)
        sub_hidden1 = F.dropout(sub_hidden1, p=0.5, training=self.training)
        output1 = self.output1(sub_hidden1)

        sub_hidden2 = self.sub_net2(hidden)
        sub_hidden2 = F.dropout(sub_hidden2, p=0.5, training=self.training)
        output2 = self.output1(sub_hidden2)

        output = torch.cat([output1, output2], 1)
        return output


def test(model, data_loader):
    model.eval()
    model.is_training = False
    with torch.no_grad():
        for arrays, torsions, output_filename in data_loader:
            torsions = torsions.to(device)
            arrays = arrays.to(device)
            pred_sincos = model(arrays[0]).squeeze(1).transpose(0, 1)

            output = np.concatenate((pred_sincos.data.cpu().numpy(), torsions.data.cpu().numpy()[0]), 0)
            np.save(os.path.join(test_output_folder, output_filename[0]), output)


if __name__ == '__main__':
    models_path = os.path.join(os.getcwd(), 'top_models')
    for model_name in os.listdir(models_path):
        if model_name[-4:] == '.pth':
            print(model_name)
            # test_output_folder = os.path.join(models_path, 'test_outputs/%s' % model_name[:-4])
            test_output_folder = os.path.join(os.getcwd(), 'comparison_test_outputs/%s' % model_name[:-4])
            pathlib.Path(test_output_folder).mkdir(parents=True, exist_ok=True)
            test_model = torch.load(os.path.join(models_path, model_name)).to(device)

            test(test_model, test_data_loader)
