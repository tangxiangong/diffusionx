from . import random
from .types import DType
from typing import Union
import numpy as np

real = Union[float, int]


class Uniform:
    def __init__(
        self,
        low: real = 0.0,
        high: real = 1.0,
        end: bool = False,
        dtype: DType = DType.Float,
    ):
        """均匀分布

        Args:
            low (real, optional): 下限. Defaults to 0.0.
            high (real, optional): 上限. Defaults to 1.0.
            end (bool, optional): 是否包含上限. Defaults to False.
            dtype (DType, optional): 数据类型. Defaults to DType.FLOAT.
        """
        self.low = low
        self.high = high
        self.end = end
        self.dtype = dtype

    def sample(self, n: int = 1) -> real | np.ndarray:
        """均匀分布随机数

        Args:
            n (int, optional): 随机数个数. Defaults to 1.

        Returns:
            real | np.ndarray: 均匀分布随机数
        """
        return random.uniform_rand(n, self.low, self.high, self.end, self.dtype)


class Normal:
    def __init__(self, mu: real = 0.0, sigma: real = 1.0):
        """正态分布

        Args:
            mu (real, optional): 均值. Defaults to 0.0.
            sigma (real, optional): 标准差. Defaults to 1.0.
        """
        self.mu = mu
        self.sigma = sigma

    def sample(self, n: int = 1) -> real | np.ndarray:
        """正态分布随机数

        Args:
            n (int, optional): 随机数个数. Defaults to 1.

        Returns:
            real | np.ndarray: 正态分布随机数
        """
        return random.normal_rand(n, self.mu, self.sigma)


class Exponential:
    def __init__(self, scale: real = 1.0):
        """指数分布

        Args:
            scale (real, optional): 尺度参数. Defaults to 1.0.
        """
        self.scale = scale

    def sample(self, n: int = 1) -> real | np.ndarray:
        """指数分布随机数

        Args:
            n (int, optional): 随机数个数. Defaults to 1.

        Returns:
            real | np.ndarray: 指数分布随机数
        """
        return random.exp_rand(n, self.scale)
