from typing import Callable

from torch import Tensor, sigmoid


def rescale(t: Tensor, lower: Tensor | float, upper: Tensor | float):
    """
    assumes t to be normalized to [0,1] and scales it into [lower, upper].
    only 1D case tested.
    """
    diff = upper - lower
    return (t * diff) + lower


def window_function(
    lower: Tensor | float, upper: Tensor | float, sigma: float
) -> Callable[..., Tensor]:
    def inner(x: Tensor):
        return sigmoid((x - lower) / sigma) * sigmoid((upper - x) / sigma)

    return inner
