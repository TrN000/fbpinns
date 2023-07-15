"""
Extended problems
"""
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, cos, log10, mean, sin
from torch.optim import Adam, LBFGS
from torch.quasirandom import SobolEngine
from torch.utils.data import DataLoader, TensorDataset

from fbpinns.net import Additive, Multiplicative, NeuralNet
from fbpinns.utils import Window, partition, rescale
from fbpinns.base import SOBOL, EXTREMA, HighFreqFb


class CumulativeApproach(HighFreqFb):
    def __init__(self, *args, **kwargs):
        super(CumulativeApproach, self).__init__(*args, **kwargs)
        # partition input space/instantiate NN
        self.model = Additive(
            # simple model
            [NeuralNet(1, 1, self.layers, neurons=self.hidden)]
            # FBPINN
            + [
                Multiplicative(
                    [
                        NeuralNet(1, 1, layers=self.layers, neurons=self.hidden),
                        Window(lower, upper, 30),
                    ]
                )
                for lower, upper in sorted(
                    partition(self.extrema[0], self.extrema[1], 30, 0.0),
                    key=lambda x: min(abs(x[0]), abs(x[1])),
                )
            ]
        )


class LogarithmicApproach(HighFreqFb):
    def __init__(self, *args, **kwargs):
        super(LogarithmicApproach, self).__init__(*args, **kwargs)
        # partition input space/instantiate NN
        self.model = Additive(
            # simple model
            [NeuralNet(1, 1, self.layers, neurons=self.hidden)]
            # FBPINN with 2 comp
            + [
                Multiplicative(
                    [
                        NeuralNet(1, 1, layers=self.layers, neurons=self.hidden),
                        Window(lower, upper, 30),
                    ]
                )
                for lower, upper in sorted(
                    partition(self.extrema[0], self.extrema[1], 2, 0.0),
                    key=lambda x: min(abs(x[0]), abs(x[1])),
                )
            ]
            # FBPINN with 4 comp
            + [
                Multiplicative(
                    [
                        NeuralNet(1, 1, layers=self.layers, neurons=self.hidden),
                        Window(lower, upper, 30),
                    ]
                )
                for lower, upper in sorted(
                    partition(self.extrema[0], self.extrema[1], 4, 0.0),
                    key=lambda x: min(abs(x[0]), abs(x[1])),
                )
            ]
            # FBPINN with 8 comp
            + [
                Multiplicative(
                    [
                        NeuralNet(1, 1, layers=self.layers, neurons=self.hidden),
                        Window(lower, upper, 30),
                    ]
                )
                for lower, upper in sorted(
                    partition(self.extrema[0], self.extrema[1], 8, 0.0),
                    key=lambda x: min(abs(x[0]), abs(x[1])),
                )
            ]
        )


def main():
    it = 1000

    fb_problem = HighFreqFb(2, 16) # 9630
    c_problem = CumulativeApproach(2, 16) # 9951
    l_problem = LogarithmicApproach(2, 16) # 4815
    breakpoint()

    fb_hist = fb_problem.fit(it)
    c_hist = c_problem.fit(it)
    l_hist = l_problem.fit(it)

    # plot
    x = torch.linspace(EXTREMA[0], EXTREMA[1], 10_000).reshape([-1, 1])

    c_sol = c_problem.model(x).detach().numpy()
    l_sol = l_problem.model(x).detach().numpy()
    f_sol = fb_problem.model(x).detach().numpy()
    actual = c_problem.exact_solution(x).detach().numpy()

    # plot
    x = torch.linspace(EXTREMA[0], EXTREMA[1], 10_000).reshape([-1, 1])
    y = l_problem.model(x).detach().numpy()
    actual = l_problem.exact_solution(x).detach().numpy()
    plt.plot(x, y, label="cumulative")
    plt.plot(x, actual, label="actual")
    plt.show()

    cols = matplotlib.colors.TABLEAU_COLORS
    keys = list(cols)

    _, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=150)

    axs[0, 0].plot(x.detach().numpy(), f_sol, color=cols[keys[0]], label="fbpinn")
    axs[0, 0].plot(x.detach().numpy(), actual, color=cols[keys[2]], label="actual")
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("f(x)")
    axs[0, 0].grid(True, which="both", ls=":")
    axs[0, 0].legend()
    axs[0, 0].set_title("FBPINN")

    axs[1, 0].plot(x.detach().numpy(), c_sol, color=cols[keys[1]], label="cumulative")
    axs[1, 0].plot(x.detach().numpy(), actual, color=cols[keys[2]], label="actual")
    axs[1, 0].set_xlabel("x")
    axs[1, 0].set_ylabel("f(x)")
    axs[1, 0].grid(True, which="both", ls=":")
    axs[1, 0].legend()
    axs[1, 0].set_title("Cumulative")

    axs[0, 1].plot(x.detach().numpy(), l_sol, color=cols[keys[1]], label="logarithmic")
    axs[0, 1].plot(x.detach().numpy(), actual, color=cols[keys[2]], label="actual")
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("f(x)")
    axs[0, 1].grid(True, which="both", ls=":")
    axs[0, 1].legend()
    axs[0, 1].set_title("Logarithmic")


    axs[1, 1].plot(np.arange(1, len(c_hist) + 1), c_hist, label="cumulative")
    axs[1, 1].plot(np.arange(1, len(l_hist) + 1), l_hist, label="logarithmic")
    axs[1, 1].plot(np.arange(1, len(fb_hist) + 1), fb_hist, label="fbpinn")
    axs[1, 1].set_xlabel("iterations")
    axs[1, 1].set_ylabel("log-loss")
    axs[1, 1].set_yscale("log")
    axs[1, 1].grid(True, which="both", ls=":")
    axs[1, 1].legend()
    axs[1, 1].set_title("Loss")

    plt.subplots_adjust(hspace=0.5)
    filename = f"./img/fb_ext{it}"
    plt.savefig(filename)
    plt.show()
