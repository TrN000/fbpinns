"""
module for neural network architectures.
Provides a fully connected network as well as
Multiplicative and Additive class, which subclas nn.ModuleList.
They apply every module in their module lists and sum/prod their results
element-wise.
"""


import torch
from torch import Tensor
import torch.nn as nn


class NeuralNet(nn.Module):
    """Copied from project A"""

    def __init__(self, input_dim: int, output_dim: int, layers: int, neurons: int):
        super(NeuralNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neurons = neurons
        self.layers = layers
        self.activation = nn.Tanh()

        if self.layers != 0:
            self.input_layer = nn.Linear(self.input_dim, self.neurons)
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(self.neurons, self.neurons) for _ in range(self.layers - 1)]
            )
            self.output_layer = nn.Linear(self.neurons, self.output_dim)

        else:
            print("Simple linear regression")
            self.linear_regression_layer = nn.Linear(self.input_dim, self.output_dim)

        torch.manual_seed(1444)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain("tanh")
                torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        if self.layers != 0:
            x = self.activation(self.input_layer(x))
            for l in self.hidden_layers:
                x = self.activation(l(x))
            return self.output_layer(x)
        else:
            return self.linear_regression_layer(x)


class Additive(nn.ModuleList):
    def __init__(self, *args):
        super(Additive, self).__init__(*args)
        # make assertions about input and output dimensions

    def forward(self, x: Tensor) -> Tensor:
        acc = torch.zeros_like(x)
        for module in self:
            acc += module(x)
        return acc


class Multiplicative(nn.ModuleList):
    def __init__(self, *args):
        super(Multiplicative, self).__init__(*args)

    def forward(self, x: Tensor) -> Tensor:
        acc = torch.ones_like(x)
        for module in self:
            acc *= module(x)
        return acc
