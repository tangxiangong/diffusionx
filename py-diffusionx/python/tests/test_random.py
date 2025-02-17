import pytest
import numpy as np
from diffusionx.random import (
    randexp,
    uniform,
    randn,
    poisson,
    stable_rand,
    skew_stable_rand,
    bool_rand,
)


def check_shape(arr: np.ndarray, expected_shape: tuple[int, ...]) -> bool:
    """检查数组形状是否符合预期

    Args:
        arr (np.ndarray): 待检查的数组
        expected_shape (tuple[int, ...]): 期望的形状

    Returns:
        bool: 形状是否符合预期
    """
    return arr.shape == expected_shape


@pytest.mark.parametrize(
    "size",
    [
        1,
        (2,),
        (2, 3),
        (2, 3, 4),
    ],
)
def test_randexp_shape(size: int | tuple[int, ...]) -> None:
    """测试指数分布随机数生成器的形状"""
    result = randexp(size=size)
    if isinstance(size, int):
        if size == 1:
            assert isinstance(result, float)
        else:
            assert isinstance(result, np.ndarray)
            assert check_shape(result, (size,))
    else:
        assert isinstance(result, np.ndarray)
        assert check_shape(result, size)


@pytest.mark.parametrize(
    "scale",
    [0.5, 1.0, 2.0],
)
def test_randexp_mean(scale: float) -> None:
    """测试指数分布随机数的均值

    通过生成大量样本并计算均值来验证
    """
    n_samples = 100000
    samples = randexp(size=n_samples, scale=scale)
    assert isinstance(samples, np.ndarray)
    assert np.abs(np.mean(samples) - scale) < 0.1


def test_randexp_invalid_params() -> None:
    """测试指数分布随机数生成器的参数验证"""
    with pytest.raises(ValueError):
        randexp(size=0)

    with pytest.raises(ValueError):
        randexp(size=-1)

    with pytest.raises(ValueError):
        randexp(scale=0)

    with pytest.raises(ValueError):
        randexp(scale=-1)


@pytest.mark.parametrize(
    "size,low,high",
    [
        (10000, 0.0, 1.0),
        (10000, -1.0, 1.0),
        (10000, 10.0, 20.0),
    ],
)
def test_uniform_range(size: int, low: float, high: float) -> None:
    """测试均匀分布随机数的范围"""
    result = uniform(size=size, low=low, high=high)
    assert isinstance(result, np.ndarray)
    assert np.all(result >= low)
    assert np.all(result <= high)
    # 检查均值是否接近理论值
    expected_mean = (low + high) / 2
    assert np.abs(np.mean(result) - expected_mean) < 0.1


@pytest.mark.parametrize(
    "size,mu,sigma",
    [
        (10000, 0.0, 1.0),
        (10000, -1.0, 2.0),
        (10000, 5.0, 0.5),
    ],
)
def test_randn_stats(size: int, mu: float, sigma: float) -> None:
    """测试正态分布随机数的统计特性"""
    result = randn(size=size, mu=mu, sigma=sigma)
    assert isinstance(result, np.ndarray)
    # 检查均值
    assert np.abs(np.mean(result) - mu) < 0.1
    # 检查标准差
    assert np.abs(np.std(result) - sigma) < 0.1


@pytest.mark.parametrize(
    "size,lambda_",
    [
        (10000, 1.0),
        (10000, 5.0),
        (10000, 10.0),
    ],
)
def test_poisson_mean(size: int, lambda_: float) -> None:
    """测试泊松分布随机数的均值"""
    result = poisson(size=size, lambda_=lambda_)
    assert isinstance(result, np.ndarray)
    # 检查均值是否接近lambda
    assert np.abs(np.mean(result) - lambda_) < 0.2


@pytest.mark.parametrize(
    "size,p",
    [
        (10000, 0.3),
        (10000, 0.5),
        (10000, 0.7),
    ],
)
def test_bool_rand_probability(size: int, p: float) -> None:
    """测试布尔随机数生成器的概率"""
    result = bool_rand(size=size, p=p)
    assert isinstance(result, np.ndarray)
    # 检查True的比例是否接近p
    assert np.abs(np.mean(result) - p) < 0.02


def test_stable_rand_basic() -> None:
    """测试稳定分布随机数的基本功能"""
    result = stable_rand(alpha=1.5, beta=0.5, size=1000)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1000,)


def test_skew_stable_rand_basic() -> None:
    """测试偏稳定分布随机数的基本功能"""
    result = skew_stable_rand(alpha=0.7, size=1000)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1000,)


@pytest.mark.parametrize(
    "size,alpha,beta,sigma,mu",
    [
        (1000, 0.7, 0.3, 1.0, 0.0),
        (1000, 1.7, 0.7, 1.0, 0.0),
    ],
)
def test_characteristic_function(
    size: int, alpha: float, beta: float, sigma: float, mu: float
) -> None:
    """
    验证经验特征函数与理论特征函数是否匹配
    """
    samples = stable_rand(alpha=alpha, beta=beta, sigma=sigma, mu=mu, size=size)
    t_values = np.linspace(-5, 5, 100)
    empirical_cf = np.mean(np.exp(1j * t_values[:, None] * samples), axis=1)

    # 理论特征函数公式
    theoretical_cf = np.exp(
        1j * mu * t_values
        - (sigma * np.abs(t_values)) ** alpha
        * (1 - 1j * beta * np.sign(t_values) * np.tan(np.pi * alpha / 2))
    )
    # 允许5%的误差
    assert np.allclose(empirical_cf, theoretical_cf, atol=0.1), "特征函数不匹配"
