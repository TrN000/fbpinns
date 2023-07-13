from typing import Callable, Tuple

from torch import Tensor, sigmoid
import torch.nn as nn


def rescale(t: Tensor, lower: Tensor | float, upper: Tensor | float):
    """
    assumes t to be normalized to [0,1] and scales it into [lower, upper].
    only 1D case tested.
    """
    diff = upper - lower
    return (t * diff) + lower


def window(
    lower: Tensor | float, upper: Tensor | float, sigma: float
) -> Callable[..., Tensor]:
    """sigma: bigger means tighter cutoff"""

    def inner(x: Tensor):
        return sigmoid((x - lower) * sigma) * sigmoid((upper - x) * sigma)

    return inner


def partition(
    lower: float, upper: float, partitions: int, overlap: float
) -> list[Tuple[float, float]]:
    # TODO: error handling and edge cases
    # distance between midpoints of the windows
    d_mid = (upper - lower) / (partitions - 1)
    # midpoints of the partition
    mids = [lower + i * d_mid for i in range(partitions)]
    # offset of the ends of the window from the midpoint
    over = (d_mid + overlap) / 2
    return [(mid - over, mid + over) for mid in mids]


class Window(nn.Module):
    """
    window function as a nn module. this is so that it can be used
    in places like `Sequential`.
    """

    def __init__(self, lower, upper, sigma):
        super(Window, self).__init__()
        self.window = window(lower, upper, sigma)

    def forward(self, x: Tensor) -> Tensor:
        return self.window(x)
