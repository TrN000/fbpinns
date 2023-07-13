from typing import Callable, Tuple

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
