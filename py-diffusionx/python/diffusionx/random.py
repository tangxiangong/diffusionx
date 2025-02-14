import numpy as np
from . import _core


def exp_rand(n: int = 1, scale: float = 1.0) -> float | np.ndarray:
    if n == 1:
        return _core.exp_rand(scale)
    else:
        return _core.exp_rands(n, scale=scale)


def uniform_rand(n: int = 1, low: float = 0.0, high: float = 1.0) -> float | np.ndarray:
    if n == 1:
        return _core.uniform_rand(low, high)
    else:
        return _core.uniform_rands(n, low=low, high=high)


def normal_rand(n: int = 1, mu: float = 0.0, sigma: float = 1.0) -> float | np.ndarray:
    if n == 1:
        return _core.normal_rand(mu, sigma)
    else:
        return _core.normal_rands(n, mu=mu, sigma=sigma)
