from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, cos, log10, mean, sin, tanh
import torch.nn as nn
from torch.optim import Adam, LBFGS
from torch.quasirandom import SobolEngine
from torch.utils.data import DataLoader, TensorDataset

from fbpinns import net
from fbpinns.utils import rescale

"""
Base Task:
    Replicate section 5.2.2/figure 7 from the paper.
"""

SOBOL = SobolEngine(1, seed=1444)

# for type hinting neural nets
ModelType = Callable[..., Tensor]


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

        # create support points
        domain = SOBOL.draw(200 * 15)
        self.simple_domain = rescale(domain, -torch.pi, torch.pi)

        self.training_set_simple = DataLoader(
            TensorDataset(self.simple_domain, self.pde(self.simple_domain)),
            batch_size=self.simple_domain.shape[0],
            shuffle=False,
        )

        # partition input space
        # instantiate NN
        self.simple: ModelType = net.NeuralNet(
            1, 1, layers=self.layers, neurons=self.hidden
        )
        self.fbpinn = nn.Sequential(
            *[
                net.NeuralNet(1, 1, layers=self.layers, neurons=self.hidden)
                for _ in range(self.subdivisions)
            ]
        )

    def exact_solution(self, x: Tensor) -> Tensor:
        return sin(self.o_1 * x) + sin(self.o_2 * x)

    def pde(self, x: Tensor) -> Tensor:
        return self.o_1 * cos(self.o_1 * x) + self.o_2 * cos(self.o_2 * x)

    def fit_simple(
        self, epochs: int, optimizer: LBFGS | Adam, verbose: bool = True
    ) -> list[float]:
        history: list[float] = []
        for epoch in range(epochs):
            if verbose:
                print(f" {epoch}".rjust(23, "#"))
            for inp, out in self.training_set_simple:

                def closure():
                    optimizer.zero_grad()

                    # calculate loss
                    inp.requires_grad = True
                    u = self.simple(inp)
                    du = torch.autograd.grad(u.sum(), inp, create_graph=True)[0]
                    pde_loss = mean(abs(du - out) ** 2)
                    bound = self.simple(Tensor([0.0]))
                    if verbose:
                        print(
                            "pde :\t",
                            log10(pde_loss).item(),
                            "bound:\t",
                            log10(bound).item(),
                        )
                    loss = log10(pde_loss + abs(bound) ** 2)

                    loss.backward()
                    history.append(loss.item())
                    return float(loss)

                optimizer.step(closure=closure)
        return history

    def fit_fb(self):
        pass

    def plot(self, history: list):
        # TODO: add option to write to image
        # TODO: make history implicit
        inputs, _ = rescale(SOBOL.draw(1000), -6, 6).sort(dim=0)
        outputs = self.simple(inputs).detach().numpy()
        actual = self.exact_solution(inputs).detach().numpy()

        _, axs = plt.subplots(2, 1, figsize=(16, 10), dpi=150)
        axs[0].plot(inputs.detach().numpy(), actual)
        axs[0].plot(inputs.detach().numpy(), outputs)
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("f(x)")
        axs[0].grid(True, which="both", ls=":")

        axs[1].plot(np.arange(1, len(history) + 1), history, label="Train Loss")
        # axs[1].xscale("log")
        axs[1].set_xlabel("iterations")
        axs[1].set_ylabel("log-loss")
        axs[1].grid(True, which="both", ls=":")

        axs[0].set_title("Measurements")
        axs[1].set_title("Solid Solution")

        plt.show()


def main():
    # TODO: use argparse to allow execution paths
    problem = SinusoidalProblem(5, 128)

    optim_lbfgs = LBFGS(
        problem.simple.parameters(),
        lr=float(0.5),
        max_iter=10000,
        max_eval=10000,
        history_size=150,
        line_search_fn="strong_wolfe",
        tolerance_change=float(1.0 * np.finfo(float).eps),
    )

    optim_adam = Adam(
        problem.simple.parameters(),
        lr=float(1e-3),
    )

    # loss_history_simple = problem.fit_simple(1000, optim_adam)
    loss_history_simple = problem.fit_simple(1, optim_lbfgs)

    problem.plot(loss_history_simple)

    problem.fit_fb()
