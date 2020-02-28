import torch
import torch.nn as nn
from npy_data_loader import DistanceWindow
from torch.utils.data import DataLoader
import os
from torch.nn import functional as F
import math


torch.cuda.set_device(0)
device = torch.device('cuda:0')

batch_size = 1
loss_function_2 = nn.MSELoss()
val_epoch = 200


if torch.cuda.is_available():
    print('GPU available!!!')
    print('MainDevice=', device)


train_name = 'nr40_Split_L1_Drop05'
save_dir = './outputs/' + train_name
val_dir = os.path.join(save_dir, 'val')

is_cross_validation = True
cross_validation_fold = 10


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


def validation(model, data_loader):
    model.eval()
    model.is_training = False
    with torch.no_grad():
        loss_sum = 0

        for arrays, torsions, output_filename in data_loader:
            torsions = torsions.to(device)
            arrays = arrays.to(device)
            sincos = torsions[0][2:]
            pred_sincos = model(arrays[0]).squeeze(1).transpose(0, 1)

            inner_error = (sincos[:2] - sincos[2:]).abs()
            weight = torch.pow(math.e, -inner_error)
            loss = torch.add(
                torch.add(
                    (torch.pow((pred_sincos[:2] - sincos[:2]).abs() + 1e-10, weight)).mean(),
                    (torch.pow((pred_sincos[2:] - sincos[2:]).abs() + 1e-10, weight)).mean()),
                torch.sqrt(loss_function_2(pred_sincos[:2], pred_sincos[2:])))

            loss_sum += float(loss)
        return loss_sum


def main():
    if is_cross_validation:
        for subset_index in range(cross_validation_fold):
            val_dataset = DistanceWindow(
                distance_window_path='/share/Data/processed/cif_190917/distance_window/',
                torsion_path='/share/Data/processed/nr40/10fold_val_subset/subset_%d/val' % subset_index)
            val_loader = DataLoader(dataset=val_dataset, pin_memory=True)

            writer = open('./outputs/%s_%d/validation_map.txt' % (train_name, subset_index), 'w')

            for epoch in range(val_epoch):
                writer.write('epoch %d\n' % epoch)
                val_model = torch.load('./outputs/%s_%d/%d_Linear.pth' % (train_name, subset_index, epoch)).to(device)

                loss_sum = validation(val_model, val_loader)
                mean_loss = loss_sum / len(val_dataset)

                writer.write('mean_val_loss=%f\n\n' % mean_loss)
                print('epoch %d, mean_val_loss=%f\n' % (epoch, mean_loss))

            writer.close()


def collect_result():
    for subset_index in range(cross_validation_fold):
        writer = open('./outputs/%s_%d/validation_map.txt' % (train_name, subset_index), 'w')
        with open('./outputs/%s_%d/val_loss.txt' % (train_name, subset_index), 'r') as file:
            lines = file.readlines()
        for i in range(len(lines)):
            if lines[i][0] == 'v':
                epoch_len = i+3
                result_index = i
                break
        epoch = 0
        for result_line in lines[result_index::epoch_len]:
            writer.write('epoch %d\n' % epoch)
            epoch += 1
            mean_loss = result_line.split('=')[1]
            writer.write('mean_val_loss=%s\n\n' % mean_loss)
            print('subset_index %d, epoch %d, mean_val_loss=%s\n' % (subset_index, epoch, mean_loss))
        writer.close()


if __name__ == '__main__':
    # main()
    collect_result()

