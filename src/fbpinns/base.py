from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, cos, log10, mean, sin
from torch.optim import Adam, LBFGS
from torch.quasirandom import SobolEngine
from torch.utils.data import DataLoader, TensorDataset

from fbpinns.net import Additive, Multiplicative, NeuralNet
from fbpinns.utils import Window, partition, rescale

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
        self.simple_domain = rescale(domain, -2 * torch.pi, 2 * torch.pi)

        self.training_set = DataLoader(
            TensorDataset(self.simple_domain, self.pde(self.simple_domain)),
            batch_size=self.simple_domain.shape[0],
            shuffle=False,
        )

        # partition input space
        # instantiate NN
        self.simple: ModelType = NeuralNet(
            1, 1, layers=self.layers, neurons=self.hidden
        )
        self.fbpinn = Additive(
            [
                Multiplicative(
                    [
                        NeuralNet(1, 1, layers=2, neurons=16),
                        Window(lower, upper, 10),
                    ]
                )
                for lower, upper in sorted(
                    partition(-2 * torch.pi, 2 * torch.pi, 30, 0.3),
                    # partition(-1, 1, 3, 0.3),
                    key=lambda x: min(abs(x[0]), abs(x[1])),
                )
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
            for inp, out in self.training_set:

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

    def fit_fb(self, epochs: int, verbose: bool = True) -> list[float]:
        history: list[float] = []
        optimizer = Adam(
            self.fbpinn.parameters(),
            lr=float(1e-3),
        )
        for epoch in range(epochs):
            if verbose:
                print(f" {epoch}".rjust(23, "#"))

            for inp, out in self.training_set:

                def closure():
                    optimizer.zero_grad()

                    # calculate loss
                    inp.requires_grad = True
                    u = self.fbpinn.forward(inp)
                    du = torch.autograd.grad(u.sum(), inp, create_graph=True)[0]
                    if epoch % 249 == 0:
                        plt.scatter(inp.detach().numpy(), u.detach().numpy())
                        plt.show()
                    pde_loss = mean(abs(du - out) ** 2)
                    bound = self.fbpinn.forward(Tensor([0.0]))
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

    def plot_fb(self, history: list):
        # TODO: add option to write to image
        # TODO: make history implicit
        inputs, _ = rescale(SOBOL.draw(1000), -6, 6).sort(dim=0)
        outputs = self.fbpinn.forward(inputs).detach().numpy()
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

    def plot(self, history: list):
        inputs, _ = rescale(SOBOL.draw(1000), -6, 6).sort(dim=0)
        outputs = self.fbpinn.forward(inputs).detach().numpy()
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

    if True:
        loss_history_simple = problem.fit_simple(
            1000,
            optimizer=Adam(
                problem.simple.parameters(),
                lr=float(1e-3),
            ),
        )
        # loss_history_simple = problem.fit_simple(1, optim_lbfgs)
        problem.plot(loss_history_simple)

    if False:
        fb_history = problem.fit_fb(500)
        problem.plot_fb(fb_history)
