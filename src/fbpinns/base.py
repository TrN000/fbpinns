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

EXTREMA = [-2 * torch.pi, 2 * torch.pi]

# for type hinting neural nets
ModelType = Callable[..., Tensor]


class HighFreqSimple:
    def __init__(
        self,
        layers: int,
        hidden: int,
        o_1: float = 1,
        o_2: float = 15,
        extrema: list[float] = EXTREMA,
    ):
        self.layers = layers
        self.hidden = hidden
        self.o_1 = o_1
        self.o_2 = o_2
        self.extrema = extrema

        # create support points
        SOBOL.reset()
        domain = SOBOL.draw(200 * 15)
        self.domain = rescale(domain, *self.extrema)

        self.training_set = DataLoader(
            TensorDataset(self.domain, self.pde(self.domain)),
            batch_size=self.domain.shape[0],
            shuffle=False,
        )

        # instantiate NN
        self.model: ModelType = NeuralNet(
            1, 1, layers=self.layers, neurons=self.hidden
        )

    def exact_solution(self, x: Tensor) -> Tensor:
        return sin(self.o_1 * x) + sin(self.o_2 * x)

    def pde(self, x: Tensor) -> Tensor:
        return self.o_1 * cos(self.o_1 * x) + self.o_2 * cos(self.o_2 * x)

    def fit(
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
                    u = self.model(inp)
                    du = torch.autograd.grad(u.sum(), inp, create_graph=True)[0]
                    pde_loss = mean(abs(du - out) ** 2)
                    bound = self.model(Tensor([0.0]))
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

    def plot(self, history: list):
        inputs, _ = rescale(SOBOL.draw(1000), -6, 6).sort(dim=0)
        outputs = self.model.forward(inputs).detach().numpy()
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


class HighFreqFb:
    def __init__(
        self,
        layers: int,
        hidden: int,
        subdivisions: int = 30,
        o_1: float = 1,
        o_2: float = 15,
        extrema: list[float] = EXTREMA,
    ):
        self.layers = layers
        self.hidden = hidden
        self.subdivisions = subdivisions
        self.o_1 = o_1
        self.o_2 = o_2
        self.extrema = extrema

        # create support points
        SOBOL.reset()
        domain = SOBOL.draw(200 * 15)
        self.domain = rescale(domain, *self.extrema)
        self.training_set = DataLoader(
            TensorDataset(self.domain, self.pde(self.domain)),
            batch_size=self.domain.shape[0],
            shuffle=False,
        )

        # partition input space/instantiate NN
        self.fbpinn = Additive(
            [
                Multiplicative(
                    [
                        NeuralNet(1, 1, layers=self.layers, neurons=self.hidden),
                        Window(lower, upper, 10),
                    ]
                )
                for lower, upper in sorted(
                    partition(self.extrema[0], self.extrema[1], 30, 0.3),
                    # partition(-1, 1, 3, 0.3),
                    key=lambda x: min(abs(x[0]), abs(x[1])),
                )
            ]
        )

    def exact_solution(self, x: Tensor) -> Tensor:
        return sin(self.o_1 * x) + sin(self.o_2 * x)

    def pde(self, x: Tensor) -> Tensor:
        return self.o_1 * cos(self.o_1 * x) + self.o_2 * cos(self.o_2 * x)

    def fit(self, epochs: int, verbose: bool = True) -> list[float]:
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

    def plot(self, history: list):
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


def main():
    # TODO: use argparse to allow execution paths

    if True:
        simple_problem = HighFreqSimple(5, 128)
        loss_history_simple = simple_problem.fit(
            1000,
            optimizer=Adam(
                simple_problem.model.parameters(),
                lr=float(1e-3),
            ),
        )
        # loss_history_simple = problem.fit_simple(1, optim_lbfgs)
        simple_problem.plot(loss_history_simple)

    if True:
        fb_problem = HighFreqFb(2, 16)
        fb_history = fb_problem.fit(500)
        fb_problem.plot(fb_history)
