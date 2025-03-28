import torch.nn as nn
import torch


class CNN_Predictor(nn.Module):
    name = 'CNN_Predictor'

    def __init__(self, features=[0], hidden_size=64, input_size=24*4, output_size=24*4):
        super(CNN_Predictor, self).__init__()
        self.features = features
        """LeNet variant from https://github.com/mit-han-lab/dlg/blob/master/models/vision.py."""
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=len(features), out_channels=hidden_size, kernel_size=5, padding=2, stride=2), nn.Sigmoid(),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5, padding=2, stride=2), nn.Sigmoid(),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5, padding=2, stride=1), nn.Sigmoid(),
        )
        self.flatten = nn.Flatten()
        temp_input = torch.zeros(1, len(features), input_size)
        temp_output = self.flatten(self.cnn(temp_input))
        flattened_size = temp_output.shape[1]
        self.fc = torch.nn.Sequential(nn.Linear(flattened_size, output_size))
        self.extra_info = {}

    def forward(self, x):
        """Transpose the input to be of shape (batch_size, n_features, seq_len)"""
        out = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        out = self.flatten(out)
        out = self.fc(out)
        return out
