import torch
import torch.nn as nn


class GRUDecoder(nn.Module):
    def __init__(self, n_features, hidden_size=64, output_length=24*4, backwards=True):
        super(GRUDecoder, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.output_length = output_length

        self.gru_cell = nn.GRUCell(n_features, hidden_size)
        self.fc = nn.Linear(hidden_size, n_features)
        self.backwards = backwards

    def forward(self, hidden, initial_input=None, targets=None, teacher_force_probability=0.0):
        outputs = []

        if initial_input is None:
            input_at_t = torch.zeros((hidden.shape[0], self.n_features), dtype=hidden.dtype, device=hidden.device)
        else:
            input_at_t = initial_input

        for i in range(self.output_length):
            hidden = self.gru_cell(input_at_t, hidden)
            input_at_t = self.fc(hidden)
            if self.backwards:
                outputs.insert(0, input_at_t.unsqueeze(1))  # Add the time step dimension
            else:
                outputs.append(input_at_t.unsqueeze(1))

            # Teacher forcing: if enabled and ground truth is available, use it as the next input
            if targets is not None and torch.rand(1).item() < teacher_force_probability:
                if self.backwards:
                    input_at_t = targets[:, -i, :]
                else:
                    input_at_t = targets[:, i, :]

        # Concatenate along the time dimension
        return torch.cat(outputs, dim=1)

class GRU_Predictor(nn.Module):
    name = 'GRU_Predictor'

    def __init__(self, features=[0], hidden_size=64, output_size=24*4, input_size=None):
        super(GRU_Predictor, self).__init__()
        self.features = features
        self.gru = nn.GRU(len(features), hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.extra_info = {}
        
    def forward(self, x, h=None, return_hidden=False):
        with torch.backends.cudnn.flags(enabled=False):
            h_x, _ = self.gru(x, h)
        x = self.fc(h_x[:, -1, :])
        if return_hidden:
            return x, h_x[:, -1, :]
        return x


class StackedGRU_Predictor(nn.Module):
    name = 'StackedGRU_Predictor'

    def __init__(self, features=[0], hidden_size=64, output_size=24*4, input_size=None):
        super(StackedGRU_Predictor, self).__init__()
        self.features = features
        self.gru1 = nn.GRU(len(features), hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.extra_info = {}

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = self.fc(x[:, -1, :])
        return x


class CNNGRU_Predictor(nn.Module):
    name = 'CNNGRU_Predictor'

    def __init__(self, features=[0], hidden_size=64, output_size=24*4, input_size=None):
        super(CNNGRU_Predictor, self).__init__()
        self.features = features
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=len(features), out_channels=hidden_size, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.extra_info = {}

    def forward(self, x):
        out = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.gru(out)
        out = self.fc(out[:, -1, :])
        return out