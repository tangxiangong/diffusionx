from . import _core
from .types import DType
from typing import Union
import numpy as np

real = Union[float, int]


def _check_all_uint(size: tuple[int, ...]) -> bool:
    return all(isinstance(i, int) and i > 0 for i in size)


def _check_uint(size: int) -> bool:
    return size > 0


def randexp(
    size: int | tuple[int, ...] = 1, scale: real = 1.0
) -> Union[float, np.ndarray]:
    """
    Exponential distribution random numbers

    Args:
        size (int | tuple[int, ...], optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.
        scale (real, optional): exponential distribution parameter, mean of the distribution. Defaults to 1.0. Positive real number.

    Returns:
        float | np.ndarray: exponential random numbers
    """
    if scale <= 0:
        raise ValueError(f"Invalid scale {scale}, expected positive real number")

    if isinstance(scale, int):
        scale = float(scale)

    if isinstance(size, int):
        if size == 1:
            return _core.exp_rand(scale)
        elif size > 1:
            return _core.exp_rands(size, scale=scale)
        else:
            raise ValueError(f"Invalid size {size}, expected positive integer")
    elif isinstance(size, tuple):
        if _check_all_uint(size):
            length = int(np.prod(size))
            arr = _core.exp_rands(length, scale=scale)
            return arr.reshape(size)
        else:
            raise ValueError(
                f"Invalid size {size}, expected tuple of positive integers"
            )
    else:
        raise ValueError(
            f"Invalid size {size}, expected positive integer or tuple of positive integers"
        )


def _uniform_float_helper(size, low, high, end):
    if isinstance(size, int):
        if size == 1:
            return _core.uniform_rand_float(float(low), float(high), end)
        elif size > 1:
            return _core.uniform_rands_float(size, float(low), float(high), end)
        else:
            raise ValueError(f"Invalid size {size}, expected positive integer")
    elif isinstance(size, tuple):
        if _check_all_uint(size):
            length = int(np.prod(size))
            arr = _core.uniform_rands_float(length, float(low), float(high), end)
            return arr.reshape(size)
        else:
            raise ValueError(
                f"Invalid size {size}, expected tuple of positive integers"
            )
    else:
        raise ValueError(
            f"Invalid size {size}, expected positive integer or tuple of positive integers"
        )


def _uniform_int_helper(size, low, high, end):
    if isinstance(size, int):
        if size == 1:
            return _core.uniform_rand_int(int(low), int(high), end)
        elif size > 1:
            return _core.uniform_rands_int(size, int(low), int(high), end)
        else:
            raise ValueError(f"Invalid size {size}, expected positive integer")
    elif isinstance(size, tuple):
        if _check_all_uint(size):
            length = int(np.prod(size))
            arr = _core.uniform_rands_int(length, int(low), int(high), end)
            return arr.reshape(size)
        else:
            raise ValueError(
                f"Invalid size {size}, expected tuple of positive integers"
            )
    else:
        raise ValueError(
            f"Invalid size {size}, expected positive integer or tuple of positive integers"
        )


def uniform(
    size: int | tuple[int, ...] = 1,
    low: real = 0.0,
    high: real = 1.0,
    end: bool = False,
    dtype: DType = DType.Float,
) -> real | np.ndarray:
    """Uniform distribution random numbers

    Args:
        size (int | tuple[int, ...], optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.
        low (real, optional): lower bound. Defaults to 0.0.
        high (real, optional): upper bound. Defaults to 1.0.
        end (bool, optional): whether to include the upper bound. Defaults to False.
        dtype (DType, optional): data type. Defaults to DType.FLOAT.

    Returns:
        real | np.ndarray: uniform random numbers
    """
    if dtype == DType.Float:
        return _uniform_float_helper(size, low, high, end)
    elif dtype == DType.Int:
        return _uniform_int_helper(size, low, high, end)
    else:
        raise ValueError(f"Invalid dtype {dtype}, expected DType.Float or DType.Int")


def randn(
    size: int | tuple[int, ...] = 1, mu: real = 0.0, sigma: real = 1.0
) -> float | np.ndarray:
    """Normal distribution random numbers

    Args:
        size (int | tuple[int, ...], optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.
        mu (real, optional): mean. Defaults to 0.0.
        sigma (real, optional): standard deviation. Defaults to 1.0. Positive real number.

    Returns:
        float | np.ndarray: normal random numbers
    """
    if sigma <= 0:
        raise ValueError(f"Invalid sigma {sigma}, expected positive real number")

    if isinstance(mu, int):
        mu = float(mu)
    if isinstance(sigma, int):
        sigma = float(sigma)

    if isinstance(size, int):
        if size == 1:
            return _core.normal_rand(mu, sigma)
        elif size > 1:
            return _core.normal_rands(size, mu, sigma)
        else:
            raise ValueError(f"Invalid size {size}, expected positive integer")
    elif isinstance(size, tuple):
        if _check_all_uint(size):
            length = int(np.prod(size))
            arr = _core.normal_rands(length, mu=mu, sigma=sigma)
            return arr.reshape(size)
        else:
            raise ValueError(
                f"Invalid size {size}, expected tuple of positive integers"
            )
    else:
        raise ValueError(
            f"Invalid size {size}, expected positive integer or tuple of positive integers"
        )


def poisson(size: int | tuple[int, ...] = 1, lambda_: real = 1.0) -> real | np.ndarray:
    """Poisson distribution random numbers

    Args:
        size (int | tuple[int, ...], optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.
        lambda_ (real, optional): Poisson distribution parameter. Defaults to 1.0. Positive real number.

    Returns:
        real | np.ndarray: Poisson random numbers
    """
    if lambda_ <= 0:
        raise ValueError(f"Invalid lambda {lambda_}, expected positive real number")

    if isinstance(lambda_, int):
        lambda_ = float(lambda_)

    if isinstance(size, int):
        if size == 1:
            return _core.poisson_rand(lambda_)
        elif size > 1:
            return _core.poisson_rands(size, lambda_=lambda_)
        else:
            raise ValueError(f"Invalid size {size}, expected positive integer")
    elif isinstance(size, tuple):
        if _check_all_uint(size):
            length = int(np.prod(size))
            arr = _core.poisson_rands(length, lambda_=lambda_)
            return arr.reshape(size)
        else:
            raise ValueError(
                f"Invalid size {size}, expected tuple of positive integers"
            )
    else:
        raise ValueError(
            f"Invalid size {size}, expected positive integer or tuple of positive integers"
        )


def stable_rand(
    alpha: real,
    beta: real,
    sigma: real = 1.0,
    mu: real = 0.0,
    size: int | tuple[int, ...] = 1,
) -> real | np.ndarray:
    """Stable distribution random numbers

    Args:
        alpha (real): stability index. Positive real number, between 0 and 2.
        beta (real): skewness parameter. Real number, between -1 and 1.
        sigma (real, optional): scale parameter. Defaults to 1.0. Positive real number.
        mu (real, optional): location parameter. Defaults to 0.0.
        size (int | tuple[int, ...], optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.
    Returns:
        real | np.ndarray: stable random numbers
    """
    if alpha <= 0 or alpha > 2:
        raise ValueError(
            f"Invalid alpha {alpha}, expected positive real number between 0 and 2"
        )
    if beta < -1 or beta > 1:
        raise ValueError(f"Invalid beta {beta}, expected real number between -1 and 1")
    if sigma <= 0:
        raise ValueError(f"Invalid sigma {sigma}, expected positive real number")

    if isinstance(alpha, int):
        alpha = float(alpha)
    if isinstance(beta, int):
        beta = float(beta)
    if isinstance(sigma, int):
        sigma = float(sigma)
    if isinstance(mu, int):
        mu = float(mu)

    if isinstance(size, int):
        if size == 1:
            return _core.stable_rand(alpha, beta, sigma, mu)
        elif size > 1:
            return _core.stable_rands(size, alpha, beta, sigma, mu)
        else:
            raise ValueError(f"Invalid size {size}, expected positive integer")
    elif isinstance(size, tuple):
        if _check_all_uint(size):
            length = int(np.prod(size))
            arr = _core.stable_rands(length, alpha, beta, sigma, mu)
            return arr.reshape(size)
        else:
            raise ValueError(
                f"Invalid size {size}, expected tuple of positive integers"
            )
    else:
        raise ValueError(
            f"Invalid size {size}, expected positive integer or tuple of positive integers"
        )


def skew_stable_rand(alpha: real, size: int | tuple[int, ...] = 1) -> real | np.ndarray:
    """Skew stable distribution random numbers

    Args:
        alpha (real): skew stable distribution parameter, stability index. Positive real number, between 0 and 1.
        size (int | tuple[int, ...], optional): shape of the output array. Defaults to 1. Positive integer or tuple of integers.

    Returns:
        real | np.ndarray: skew stable random numbers
    """
    if alpha <= 0 or alpha > 1:
        raise ValueError(
            f"Invalid alpha {alpha}, expected positive real number between 0 and 1"
        )

    if isinstance(alpha, int):
        alpha = float(alpha)

    if isinstance(size, int):
        if size == 1:
            return _core.skew_stable_rand(alpha)
        elif size > 1:
            return _core.skew_stable_rands(size, alpha)
        else:
            raise ValueError(f"Invalid size {size}, expected positive integer")
    elif isinstance(size, tuple):
        if _check_all_uint(size):
            length = int(np.prod(size))
            arr = _core.skew_stable_rands(length, alpha)
            return arr.reshape(size)
        else:
            raise ValueError(
                f"Invalid size {size}, expected tuple of positive integers"
            )
    else:
        raise ValueError(
            f"Invalid size {size}, expected positive integer or tuple of positive integers"
        )


def bool_rand(size: tuple[int, ...] | int = 1, p: real = 0.5) -> bool | np.ndarray:
    """Boolean random numbers

    Args:
        size (tuple[int, ...] | int, optional): shape of the output array. Defaults to 1.
        p (real, optional): probability of True. Defaults to 0.5.

    Returns:
        bool | np.ndarray: boolean random numbers
    """
    if p < 0 or p > 1:
        raise ValueError(f"Invalid p {p}, expected real number between 0 and 1")

    if isinstance(size, int):
        if size == 1:
            return _core.bool_rand(p)
        elif size > 1:
            return _core.bool_rands(size, p=p)
        else:
            raise ValueError(f"Invalid size {size}, expected positive integer")
    elif isinstance(size, tuple):
        if _check_all_uint(size):
            length = int(np.prod(size))
            arr = _core.bool_rands(length, p=p)
            return arr.reshape(size)
        else:
            raise ValueError(
                f"Invalid size {size}, expected tuple of positive integers"
            )
    else:
        raise ValueError(
            f"Invalid size {size}, expected positive integer or tuple of positive integers"
        )
