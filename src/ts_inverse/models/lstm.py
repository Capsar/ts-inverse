import torch.nn as nn

class LSTM_Predictor(nn.Module):
    name = 'LSTM_Predictor'

    def __init__(self, features=[0], hidden_size=64, output_size=24*4):
        super(LSTM_Predictor, self).__init__()
        self.features = len(features)
        self.lstm = nn.LSTM(len(features), hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.extra_info = {}

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


class StackedLSTM_Predictor(nn.Module):
    name = 'StackedLSTM_Predictor'

    def __init__(self, features=[0], hidden_size=64, output_size=24*4):
        super(StackedLSTM_Predictor, self).__init__()
        self.features = len(features)
        self.lstm1 = nn.LSTM(len(features), hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.extra_info = {}

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :])
        return x


class CNNLSTM_Predictor(nn.Module):
    name = 'CNNLSTM_Predictor'

    def __init__(self, features=[0], hidden_size=64, output_size=24*4):
        super(CNNLSTM_Predictor, self).__init__()
        self.features = features
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=len(features), out_channels=hidden_size, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.extra_info = {}

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out