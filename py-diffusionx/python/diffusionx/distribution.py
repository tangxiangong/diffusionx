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
        """Uniform distribution

        Args:
            low (real, optional): lower bound. Defaults to 0.0.
            high (real, optional): upper bound. Defaults to 1.0.
            end (bool, optional): whether to include the upper bound. Defaults to False.
            dtype (DType, optional): data type. Defaults to DType.FLOAT.
        """
        self.low = low
        self.high = high
        self.end = end
        self.dtype = dtype

    def sample(self, n: int = 1) -> real | np.ndarray:
        """Uniform distribution random numbers

        Args:
            n (int, optional): number of random numbers. Defaults to 1. Positive integer.

        Returns:
            real | np.ndarray: uniform random numbers
        """
        return random.uniform(n, self.low, self.high, self.end, self.dtype)


class Normal:
    def __init__(self, mu: real = 0.0, sigma: real = 1.0):
        """Normal distribution

        Args:
            mu (real, optional): mean. Defaults to 0.0.
            sigma (real, optional): standard deviation. Defaults to 1.0. Positive real number.
        """
        self.mu = mu
        self.sigma = sigma

    def sample(self, n: int = 1) -> real | np.ndarray:
        """Normal distribution random numbers

        Args:
            n (int, optional): number of random numbers. Defaults to 1. Positive integer.

        Returns:
            real | np.ndarray: normal random numbers
        """
        return random.randn(n, self.mu, self.sigma)

    def __neg__(self):
        return Normal(self.mu, self.sigma)

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Normal(self.mu + other, self.sigma)
        else:
            raise ValueError("Invalid operand type")

    def __radd__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Normal(other + self.mu, self.sigma)
        else:
            raise ValueError("Invalid operand type")

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Normal(self.mu * other, self.sigma * other)
        else:
            raise ValueError("Invalid operand type")

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Normal(other * self.mu, other * self.sigma)
        else:
            raise ValueError("Invalid operand type")


class Exponential:
    def __init__(self, scale: real = 1.0):
        """Exponential distribution

        Args:
            scale (real, optional): scale parameter. Defaults to 1.0. Positive real number.
        """
        self.scale = scale

    def sample(self, n: int = 1) -> real | np.ndarray:
        """Exponential distribution random numbers

        Args:
            n (int, optional): number of random numbers. Defaults to 1. Positive integer.

        Returns:
            real | np.ndarray: exponential random numbers
        """
        return random.randexp(n, self.scale)


class Poisson:
    def __init__(self, lambda_: real = 1.0):
        """Poisson distribution

        Args:
            lambda_ (real, optional): Poisson distribution parameter, mean of the distribution. Defaults to 1.0. Positive real number.
        """
        self.lambda_ = lambda_

    def sample(self, n: int = 1) -> real | np.ndarray:
        """Poisson distribution random numbers

        Args:
            n (int, optional): number of random numbers. Defaults to 1. Positive integer.

        Returns:
            real | np.ndarray: Poisson distribution random numbers
        """
        return random.poisson(n, self.lambda_)


class Stable:
    def __init__(self, alpha: real, beta: real, sigma: real, mu: real):
        """Stable distribution

        Args:
            alpha (real): stability index. Positive real number, between 0 and 2.
            beta (real): skewness parameter. Real number, between -1 and 1.
            sigma (real): scale parameter. Positive real number.
            mu (real): location parameter. Real number.
        """
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.mu = mu
        self.symm: bool = False
        self.skewed: bool = False
        self.std: bool = False

    @classmethod
    def symmetric(cls, alpha: real):
        """Symmetric stable distribution

        Args:
            alpha (real): stability index. Positive real number, between 0 and 2.
        """
        result = cls(alpha, 0.0, 1.0, 0.0)
        result.symm = True
        return result

    @classmethod
    def skew(cls, alpha: real):
        """Skewed stable distribution

        Args:
            alpha (real): stability index. Positive real number, between 0 and 1.
        """
        result = cls(alpha, 1.0, 1.0, 0.0)
        result.skewed = True
        return result

    @classmethod
    def standard(cls, alpha: real, beta: real):
        """Standard stable distribution

        Args:
            alpha (real): stability index. Positive real number, between 0 and 2.
            beta (real): skewness parameter. Real number, between -1 and 1.
        """
        result = cls(alpha, beta, 1.0, 0.0)
        result.std = True
        return result

    def sample(self, n: int = 1) -> real | np.ndarray:
        """Stable distribution random numbers

        Args:
            n (int, optional): number of random numbers. Defaults to 1. Positive integer.

        Returns:
            real | np.ndarray: stable random numbers
        """
        if self.skewed:
            return random.skew_stable_rand(self.alpha, n)
        else:
            return random.stable_rand(self.alpha, self.beta, self.sigma, self.mu, n)

    def __neg__(self):
        return Stable(self.alpha, -self.beta, self.sigma, -self.mu)

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Stable(self.alpha, self.beta, self.sigma, self.mu + other)
        else:
            raise ValueError("Invalid operand type")

    def __radd__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Stable(self.alpha, self.beta, self.sigma, other + self.mu)
        else:
            raise ValueError("Invalid operand type")

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Stable(self.alpha, self.beta, self.sigma * other, self.mu)
        else:
            raise ValueError("Invalid operand type")

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Stable(self.alpha, self.beta, other * self.sigma, self.mu)
        else:
            raise ValueError("Invalid operand type")


class SkewStable(Stable):
    def __init__(self, alpha: real):
        super().__init__(alpha, 1.0, 1.0, 0.0)
        self.skewed = True

    def sample(self, n: int = 1) -> real | np.ndarray:
        return random.skew_stable_rand(self.alpha, n)
