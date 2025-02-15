from . import _core
from .types import DType
from typing import Union
import numpy as np

real = Union[float, int]


def randexp(n: int = 1, scale: real = 1.0) -> Union[float, np.ndarray]:
    """
    Exponential distribution random numbers

    Args:
        n (int, optional): number of random numbers. Defaults to 1. Positive integer.
        scale (real, optional): exponential distribution parameter, mean of the distribution. Defaults to 1.0. Positive real number.

    Returns:
        float | np.ndarray: exponential random numbers
    """
    if isinstance(scale, int):
        scale = float(scale)

    if n == 1:
        return _core.exp_rand(scale)
    else:
        return _core.exp_rands(n, scale=scale)


def uniform(
    n: int = 1,
    low: real = 0.0,
    high: real = 1.0,
    end: bool = False,
    dtype: DType = DType.Float,
) -> real | np.ndarray:
    """Uniform distribution random numbers

    Args:
        n (int, optional): number of random numbers. Defaults to 1. Positive integer.
        low (real, optional): lower bound. Defaults to 0.0.
        high (real, optional): upper bound. Defaults to 1.0.
        end (bool, optional): whether to include the upper bound. Defaults to False.
        dtype (DType, optional): data type. Defaults to DType.FLOAT.

    Returns:
        real | np.ndarray: uniform random numbers
    """
    if n == 1:
        if dtype == DType.Float:
            return _core.uniform_rand_float(float(low), float(high), end)
        else:
            return _core.uniform_rand_int(int(low), int(high), end)
    else:
        if dtype == DType.Float:
            return _core.uniform_rands_float(n, float(low), float(high), end)
        else:
            return _core.uniform_rands_int(n, int(low), int(high), end)


def randn(n: int = 1, mu: real = 0.0, sigma: real = 1.0) -> float | np.ndarray:
    """Normal distribution random numbers

    Args:
        n (int, optional): number of random numbers. Defaults to 1. Positive integer.
        mu (real, optional): mean. Defaults to 0.0.
        sigma (real, optional): standard deviation. Defaults to 1.0. Positive real number.

    Returns:
        float | np.ndarray: normal random numbers
    """
    if isinstance(mu, int):
        mu = float(mu)
    if isinstance(sigma, int):
        sigma = float(sigma)

    if n == 1:
        return _core.normal_rand(mu, sigma)
    else:
        return _core.normal_rands(n, mu=mu, sigma=sigma)


def poisson(n: int = 1, lambda_: real = 1.0) -> real | np.ndarray:
    """Poisson distribution random numbers

    Args:
        n (int, optional): number of random numbers. Defaults to 1. Positive integer.
        lambda_ (real, optional): Poisson distribution parameter. Defaults to 1.0. Positive real number.

    Returns:
        real | np.ndarray: Poisson random numbers
    """
    if isinstance(lambda_, int):
        lambda_ = float(lambda_)

    if n == 1:
        return _core.poisson_rand(lambda_)
    else:
        return _core.poisson_rands(n, lambda_=lambda_)


def stable_rand(
    alpha: real, beta: real, sigma: real = 1.0, mu: real = 0.0, n: int = 1
) -> real | np.ndarray:
    """Stable distribution random numbers

    Args:
        alpha (real): stability index. Positive real number, between 0 and 2.
        beta (real): skewness parameter. Real number, between -1 and 1.
        sigma (real, optional): scale parameter. Defaults to 1.0. Positive real number.
        mu (real, optional): location parameter. Defaults to 0.0.
        n (int, optional): number of random numbers. Defaults to 1. Positive integer.
    Returns:
        real | np.ndarray: stable random numbers
    """
    if isinstance(alpha, int):
        alpha = float(alpha)
    if isinstance(beta, int):
        beta = float(beta)
    if isinstance(sigma, int):
        sigma = float(sigma)
    if isinstance(mu, int):
        mu = float(mu)

    if n == 1:
        return _core.stable_rand(alpha, beta, sigma, mu)
    else:
        return _core.stable_rands(n, alpha, beta, sigma, mu)


def skew_stable_rand(alpha: real, n: int = 1) -> real | np.ndarray:
    """Skew stable distribution random numbers

    Args:
        alpha (real): skew stable distribution parameter, stability index. Positive real number, between 0 and 1.
        n (int, optional): number of random numbers. Defaults to 1. Positive integer.

    Returns:
        real | np.ndarray: skew stable random numbers
    """
    if isinstance(alpha, int):
        alpha = float(alpha)

    if n == 1:
        return _core.skew_stable_rand(alpha)
    else:
        return _core.skew_stable_rands(n, alpha)


def bool_rand(n: int = 1, p: real = 0.5) -> bool | np.ndarray:
    """Boolean random numbers

    Args:
        n (int, optional): number of random numbers. Defaults to 1. Positive integer.
        p (real, optional): probability of True. Defaults to 0.5.

    Returns:
        bool | np.ndarray: boolean random numbers
    """
    if n == 1:
        return _core.bool_rand(p)
    else:
        return _core.bool_rands(n, p=p)
