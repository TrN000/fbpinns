from torch import Tensor


def rescale(t: Tensor, lower: Tensor, upper: Tensor):
    """
    assumes t to be normalized to [0,1].
    only 1D case tested.
    """
    diff = upper - lower
    return (t * diff) + lower
