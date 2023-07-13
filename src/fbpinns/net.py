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
