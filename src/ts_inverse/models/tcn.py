import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

TORCH_ACTIVATIONS = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
}


def calculate_receptive_field(num_levels, kernel_size, dilation_factor):
    return 1 + 2 * (kernel_size - 1) * (1 - dilation_factor ** num_levels) // (1 - dilation_factor)


def calculate_num_levels(seq_length, kernel_size, dilation_factor):
    num_levels = 1
    current_receptive_field = 1 + 2 * (kernel_size - 1) * (1 - dilation_factor ** num_levels) // (1 - dilation_factor)
    while current_receptive_field < seq_length:
        num_levels += 1
        current_receptive_field = 1 + 2 * (kernel_size - 1) * (1 - dilation_factor ** num_levels) // (1 - dilation_factor)
    return num_levels


def calculate_kernel_size(seq_length, num_levels, dilation_factor):
    kernel_size = 2
    current_receptive_field = 1 + 2 * (kernel_size - 1) * (1 - dilation_factor ** num_levels) // (1 - dilation_factor)
    while current_receptive_field < seq_length:
        kernel_size += 1
        current_receptive_field = 1 + 2 * (kernel_size - 1) * (1 - dilation_factor ** num_levels) // (1 - dilation_factor)
    return kernel_size


def calculate_dilation_factor(seq_length, num_levels, kernel_size):
    dilation_factor = 2
    current_receptive_field = 1 + 2 * (kernel_size - 1) * (1 - dilation_factor ** num_levels) // (1 - dilation_factor)
    while current_receptive_field < seq_length:
        dilation_factor += 1
        current_receptive_field = 1 + 2 * (kernel_size - 1) * (1 - dilation_factor ** num_levels) // (1 - dilation_factor)
    return dilation_factor


class MaskDropout(nn.Dropout):
    """
        Source: https://github.com/dAI-SY-Group/DropoutInversionAttack/blob/main/src/models/modules.py
        Modified to always track the mask the first time forward is called when use_mask is false.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p, inplace)
        self.do_mask = None
        self.mask_shape = None
        self.use_mask = False
        self.track_mask = True # In original this is False
        self.p = p

    def forward(self, x):
        if self.use_mask and self.do_mask is not None and not self.track_mask:
            return x * self.do_mask * (1/(1-self.p))
        else:
            x = super().forward(x)
            if self.track_mask:
                self.do_mask = (x != 0)
                self.mask_shape = self.do_mask.shape
                self.track_mask = False
            return x


class Chomp1d(nn.Module):
    """
        Source: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
        Source: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2,
                 activation='relu', use_weight_norm=True, init_weights=True):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        if use_weight_norm:
            self.conv1 = weight_norm(self.conv1)
        self.chomp1 = Chomp1d(padding)
        self.activation1 = TORCH_ACTIVATIONS[activation]()
        self.dropout1 = MaskDropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        if use_weight_norm:
            self.conv2 = weight_norm(self.conv2)
        self.chomp2 = Chomp1d(padding)
        self.activation2 = TORCH_ACTIVATIONS[activation]()
        self.dropout2 = MaskDropout(dropout)

        self.net = nn.ModuleList([self.conv1, self.chomp1, self.activation1, self.dropout1,
                                  self.conv2, self.chomp2, self.activation2, self.dropout2])

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.final_activation = TORCH_ACTIVATIONS[activation]()

        if init_weights:
            self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = x
        for layer in self.net:
            out = layer(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_activation(out + res)


class TCN(nn.Module):
    """
        Source: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    """

    def __init__(self, n_features, seq_length, hidden_size, num_levels=0, kernel_size=0, dilation_factor=0,
                 dropout=0.2, activation='relu', use_weight_norm=True, init_weights=True):
        super(TCN, self).__init__()
        layers = []

        if (num_levels == 0 and (kernel_size == 0 or dilation_factor == 0)) or (kernel_size == 0 and dilation_factor == 0):
            raise ValueError('Of num_levels, kernel_size and dilation_factor, at least two must be specified.')

        if num_levels == 0:
            num_levels = calculate_num_levels(seq_length, kernel_size, dilation_factor)
            # print(f'Number of levels: {num_levels} for a sequence length of {seq_length}')

        if kernel_size == 0:
            kernel_size = calculate_kernel_size(seq_length, num_levels, dilation_factor)
            # print(f'Kernel size: {kernel_size} for a sequence length of {seq_length}')

        if dilation_factor == 0:
            dilation_factor = calculate_dilation_factor(seq_length, num_levels, kernel_size)
            # print(f'Dilation factor: {dilation_factor} for a sequence length of {seq_length}')

        current_receptive_field = 1 + 2 * (kernel_size - 1) * (1 - dilation_factor ** num_levels) // (1 - dilation_factor)
        # print(f'Receptive field is {current_receptive_field}')

        num_channels = [hidden_size] * num_levels
        for i in range(num_levels):
            dilation_size = dilation_factor ** i
            in_channels = n_features if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout,
                                     activation=activation, use_weight_norm=use_weight_norm, init_weights=init_weights)]
        self.network = nn.ModuleList(layers)

        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.dilation_factor = dilation_factor
        self.receptive_field = current_receptive_field

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x


class TCN_Predictor(nn.Module):
    name = 'TCN_Predictor'

    def __init__(self, features=[0], hidden_size=64, input_size=24*4, output_size=24*4, num_levels=0, kernel_size=0, dilation_factor=0,
                 dropout=0.1, activation='relu', use_weight_norm=True, init_weights=True):
        super(TCN_Predictor, self).__init__()
        self.features = features
        self.tcn = TCN(n_features=len(features), seq_length=input_size, hidden_size=hidden_size, num_levels=num_levels,
                       kernel_size=kernel_size, dilation_factor=dilation_factor, dropout=dropout, activation=activation,
                       use_weight_norm=use_weight_norm, init_weights=init_weights)
        self.fc = nn.Linear(hidden_size, output_size)
        self.extra_info = {
            'num_levels': self.tcn.num_levels,
            'kernel_size': self.tcn.kernel_size,
            'dilation_factor': self.tcn.dilation_factor,
            'receptive_field': self.tcn.receptive_field,
        }

    def forward(self, x):
        out = self.tcn(x.transpose(1, 2))
        out = self.fc(out[:, :, -1])
        return out

    def set_use_dropout_mask(self, use_mask):
        for module in self.modules():
            if module.__class__.__name__ == 'MaskDropout':
                module.use_mask = use_mask

    def set_track_dropout_mask(self, track_mask):
        for module in self.modules():
            if module.__class__.__name__ == 'MaskDropout':
                module.track_mask = track_mask

    def get_dropout_layers(self):
        return [module for module in self.modules() if module.__class__.__name__ == 'MaskDropout']

    def init_dropout_masks(self, device='cpu', type='bernoulli'):
        """
        Initialize the dropout masks for optimization.
        """
        def init_mask(shape, p):
            if type == 'bernoulli':
                return torch.bernoulli(torch.ones(shape)*(1-p)).detach().to(device).requires_grad_(True)
            elif type == 'halves':
                return (torch.ones(shape)*0.5).detach().to(device).requires_grad_(True)
            elif type == 'uniform':
                return torch.rand(shape).detach().to(device).requires_grad_(True)
            elif type == 'p':
                return (torch.ones(shape)*p).detach().to(device).requires_grad_(True)
            elif type == '1-p':
                return (torch.ones(shape)*(1-p)).detach().to(device).requires_grad_(True)
            else:
                raise ValueError(f'Unknown dropout mask initialization type: {type}')
            
        self.set_use_dropout_mask(True)
        for dropout_layer in self.get_dropout_layers():
            dropout_layer.do_mask = init_mask(dropout_layer.mask_shape, dropout_layer.p)
        return [dropout_layer.do_mask for dropout_layer in self.get_dropout_layers()]
    
    def clamp_dropout_layers(self, min=0.0, max=1.0):
        for dropout_layer in self.get_dropout_layers():
            if dropout_layer.p > 0.0:
                dropout_layer.do_mask.data.clamp_(min, max)


class TCN_Predictor_Flatten(nn.Module):
    name = 'TCN_Predictor_Flatten'

    def __init__(self, features=[0], hidden_size=64, input_size=24*4, output_size=24*4, num_levels=0, kernel_size=0, dilation_factor=0,
                 dropout=0.1, activation='relu', use_weight_norm=True, init_weights=True):
        super(TCN_Predictor, self).__init__()
        self.features = features
        self.tcn = TCN(n_features=len(features), seq_length=input_size, hidden_size=hidden_size, num_levels=num_levels,
                       kernel_size=kernel_size, dilation_factor=dilation_factor, dropout=dropout, activation=activation,
                       use_weight_norm=use_weight_norm, init_weights=init_weights)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size*input_size, output_size)
        self.extra_info = {
            'num_levels': self.tcn.num_levels,
            'kernel_size': self.tcn.kernel_size,
            'dilation_factor': self.tcn.dilation_factor,
            'receptive_field': self.tcn.receptive_field,
        }

    def forward(self, x):
        out = self.tcn(x.transpose(1, 2))
        out = self.flatten(out)
        out = self.fc(out)
        return out
