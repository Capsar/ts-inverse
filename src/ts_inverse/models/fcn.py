import torch.nn as nn

class FCN_Predictor(nn.Module):
    name = 'FCN_Predictor'

    def __init__(self, features=[0], hidden_size=64, input_size=24*4, output_size=24*4):
        super(FCN_Predictor, self).__init__()
        self.features = features
        self.input_fc = nn.Sequential(
            nn.Linear(len(features) * input_size, hidden_size), nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size), nn.Sigmoid()
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.extra_info = {}

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.input_fc(x)
        x = self.fc(x)
        return x