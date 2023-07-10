import torch
import torch.nn as nn
from torch import Tensor, sin, cos, tanh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Base Task:
    Replicate figure 7 from the paper.
"""


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
                [nn.Linear(self.neurons, self.neurons) for _ in range(layers - 1)]
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

    def forward(self, x):
        if self.layers != 0:
            x = self.activation(self.input_layer(x))
            for l in self.hidden_layers:
                x = self.activation(l(x))
            return self.output_layer(x)
        else:
            return self.linear_regression_layer(x)


class SinusoidalProblem:
    def __init__(self, o_1: float = 1, o_2: float = 15):
        self.o_1 = o_1
        self.o_2 = o_2

        # partition input space
        # create support points
        # instantiate NN

    def exact_solution(self, x: Tensor):
        return sin(self.o_1 * x) + sin(self.o_2 * x)

    def fit(self):
        pass

    def plot(self):
        pass


def main():
    problem = SinusoidalProblem()
    problem.fit()
    problem.plot()
