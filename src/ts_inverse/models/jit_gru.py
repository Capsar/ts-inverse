# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.jit as jit
import torch.nn as nn
from torch.nn import Parameter
from typing import List, Optional, Tuple
from torch import Tensor
import math


class JitGRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(JitGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        x = x.view(-1, x.size(1))
        x_results = torch.mm(x, self.weight_ih.t()) + self.bias_ih
        h_results = torch.mm(hidden, self.weight_hh.t()) + self.bias_hh

        i_r, i_z, i_n = x_results.chunk(3, 1)
        h_r, h_z, h_n = h_results.chunk(3, 1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)

        return n - torch.mul(n, z) + torch.mul(z, hidden), z.detach()


class JitGRULayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(JitGRULayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        inputs = x.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])

        for i in range(len(inputs)):
            hidden, _ = self.cell(inputs[i], hidden)
            outputs += [hidden]

        return torch.stack(outputs), hidden


class JitGRU(jit.ScriptModule):
    __constants__ = ['hidden_size', 'num_layers', 'batch_first']

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, bias=True):
        super(JitGRU, self).__init__()
        # The following are not implemented.
        assert bias

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        if num_layers == 1:
            self.layers = nn.ModuleList([JitGRULayer(JitGRUCell, input_size, hidden_size)])
        else:
            self.layers = nn.ModuleList([JitGRULayer(JitGRUCell, input_size, hidden_size)] + [JitGRULayer(JitGRUCell, hidden_size, hidden_size)
                                                                                              for _ in range(num_layers - 1)])

    @jit.script_method
    def forward(self, x, h=None):
        # type: (Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        output_states = jit.annotate(List[Tensor], [])

        # Handle batch_first cases
        if self.batch_first:
            x = x.permute(1, 0, 2)

        if h is None:
            h = torch.zeros(self.num_layers, x.shape[1], self.hidden_size, dtype=x.dtype, device=x.device)

        output = x
        i = 0

        for rnn_layer in self.layers:
            output, hidden = rnn_layer(output, h[i])
            output_states += [hidden]
            i += 1

        # Don't forget to handle batch_first cases for the output too!
        if self.batch_first:
            output = output.permute(1, 0, 2)

        return output, torch.stack(output_states)


class JitGRU_Predictor(nn.Module):
    name = 'JitGRU_Predictor'

    def __init__(self, features=[0], hidden_size=64, output_size=24*4, input_size=None):
        super(JitGRU_Predictor, self).__init__()
        self.features = features
        self.gru = JitGRU(len(features), hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.extra_info = {}

    def forward(self, x, h=None, return_hidden=False):
        h_x, _ = self.gru(x, h)

        x = self.fc(h_x[:, -1, :])
        if return_hidden:
            return x, h_x[:, -1, :]
        return x


class CNNJitGRU_Predictor(nn.Module):
    name = 'CNNJitGRU_Predictor'

    def __init__(self, features=[0], hidden_size=64, output_size=24*4, input_size=None):
        super(CNNJitGRU_Predictor, self).__init__()
        self.features = features
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=len(features), out_channels=hidden_size, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.gru = JitGRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.extra_info = {}

    def forward(self, x):
        out = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.gru(out)
        out = self.fc(out[:, -1, :])
        return out


class JitSeq2Seq_Predictor(jit.ScriptModule):
    name = 'JitSeq2Seq_Predictor'
    __constants__ = ['output_length']

    def __init__(self, features, hidden_size=64, output_size=24*4, input_size=None):
        super(JitSeq2Seq_Predictor, self).__init__()
        self.encoder_gru = JitGRU(len(features), hidden_size, batch_first=True)
        self.decoder_gru = JitGRU(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.features = features
        self.output_length = output_size
        self.extra_info = {}


    @jit.script_method
    def forward(self, src: Tensor, trg: Optional[Tensor] = None, teacher_forcing_ratio: float = 0.5) -> Tensor:
        # src: (batch_size, src_seq_len, input_dim)
        # trg: (batch_size, trg_seq_len, output_dim)
        batch_size = src.size(0)
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, self.output_length, 1).to(src.device)

        # Encode the source sequence
        _, hidden = self.encoder_gru(src)

        # First input to the decoder is last input
        input = src[:, -1, :]

        for t in range(self.output_length):
            # Insert input into decoder
            output, hidden = self.decoder_gru(input.unsqueeze(1), hidden)
            output = self.fc(output).squeeze(1)
            outputs[:, t, :] = output

            # Decide if we are going to use teacher forcing or not
            if trg is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input = trg[:, t, :]
            else:
                input = output

        return outputs
    
class JitGRUDecoder(jit.ScriptModule):
    __constants__ = ['hidden_size', 'output_length', 'output_size']

    def __init__(self, n_features, hidden_size=64, output_length=24*4):
        super(JitGRUDecoder, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.output_length = output_length

        self.gru_cell = JitGRUCell(n_features, hidden_size)
        self.fc = nn.Linear(hidden_size, n_features)

    @jit.script_method
    def forward(
            self, hidden: Tensor, initial_input: Optional[Tensor] = None, targets: Optional[Tensor] = None, teacher_force_probability: float = 0.0) -> Tensor:
        outputs = torch.jit.annotate(List[Tensor], [])

        if initial_input is None:
            input_at_t = torch.zeros((hidden.shape[0], self.n_features), dtype=hidden.dtype, device=hidden.device)
        else:
            input_at_t = initial_input

        for i in range(self.output_length):
            hidden = self.gru_cell(input_at_t, hidden)
            input_at_t = self.fc(hidden)
            outputs.insert(0, input_at_t.unsqueeze(1))  # Add the time step dimension

            # If teacher forcing is enabled, we will use the ground truth as the input at the next time step
            if targets is not None and torch.rand(1).item() < teacher_force_probability:
                input_at_t = targets[:, i, :]

        # return torch.cat(outputs, dim=1)
        return torch.stack(outputs), hidden
