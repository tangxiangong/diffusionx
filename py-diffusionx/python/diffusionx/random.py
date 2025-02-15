from . import _core
from .types import DType
from typing import Union
import numpy as np

real = Union[float, int]


def randexp(n: int = 1, scale: real = 1.0) -> Union[float, np.ndarray]:
    """
    指数分布随机数

    Args:
        n (int, optional): 随机数个数. Defaults to 1.
        scale (real, optional): 指数分布参数. Defaults to 1.0.

    Returns:
        float | np.ndarray: 指数分布随机数
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
    """均匀分布随机数

    Args:
        n (int, optional): 随机数个数. Defaults to 1.
        low (real, optional): 均匀分布下限. Defaults to 0.0.
        high (real, optional): 均匀分布上限. Defaults to 1.0.
        end (bool, optional): 是否包含上限. Defaults to False.
        dtype (DType, optional): 数据类型. Defaults to DType.FLOAT.

    Returns:
        real | np.ndarray: 均匀分布随机数
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
    """正态分布随机数

    Args:
        n (int, optional): 随机数个数. Defaults to 1.
        mu (real, optional): 正态分布均值. Defaults to 0.0.
        sigma (real, optional): 正态分布标准差. Defaults to 1.0.

    Returns:
        float | np.ndarray: 正态分布随机数
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
    """泊松分布随机数

    Args:
        n (int, optional): 随机数个数. Defaults to 1.
        lambda_ (real, optional): 泊松分布参数. Defaults to 1.0.

    Returns:
        real | np.ndarray: 泊松分布随机数
    """
    if isinstance(lambda_, int):
        lambda_ = float(lambda_)

    if n == 1:
        return _core.poisson_rand(lambda_)
    else:
        return _core.poisson_rands(n, lambda_=lambda_)
