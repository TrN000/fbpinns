import torch
import torch.nn as nn
from torch import Tensor, sin, cos, tanh
import matplotlib.pyplot as plt
import numpy as np
from fbpinns import net
from torch.quasirandom import SobolEngine

"""
Base Task:
    Replicate section 5.2.2/figure 7 from the paper.
"""

SOBOL = SobolEngine(1, seed=1444)


class SinusoidalProblem:
    def __init__(
        self,
        layers: int,
        hidden: int,
        o_1: float = 1,
        o_2: float = 15,
        subdivisions: int = 30,
    ):
        self.layers = layers
        self.hidden = hidden
        self.o_1 = o_1
        self.o_2 = o_2
        self.subdivisions = subdivisions

        # partition input space
        # create support points
        # instantiate NN
        self.simple = net.NeuralNet(1, 1, layers=self.layers, neurons=self.hidden)
        self.fbpinn = nn.Sequential(
            *[
                net.NeuralNet(1, 1, layers=self.layers, neurons=self.hidden)
                for _ in range(self.subdivisions)
            ]
        )

    def exact_solution(self, x: Tensor):
        return sin(self.o_1 * x) + sin(self.o_2 * x)

    def fit_simple(self):
        pass

    def fit_fb(self):
        pass

    def plot(self):
        pass


def main():
    problem = SinusoidalProblem()
    problem.fit_simple()
    problem.fit_fb()
    problem.plot()
