"""Lévy Stable Distribution 的 Numba 实现

这个模块提供了 Lévy Stable Distribution 的高性能实现，使用 Numba 进行加速。
该实现基于以下参数：
- α (alpha): 稳定性指数，取值范围 (0, 2]
- β (beta): 偏度参数，取值范围 [-1, 1]
- σ (sigma): 尺度参数，取值范围 (0, ∞)
- μ (mu): 位置参数，取值范围 (-∞, ∞)
"""

from typing import Optional
import numpy as np
import numba as nb  # type: ignore


@nb.njit
def _sample_standard_alpha(
    alpha: float, beta: float, seed: Optional[int] = None
) -> float:
    """生成标准 α-stable 分布的一个样本（α ≠ 1）"""
    if seed is not None:
        np.random.seed(seed)
    half_pi = np.pi / 2.0
    tmp = beta * np.tan(alpha * half_pi)
    v = np.random.uniform(-half_pi, half_pi)
    w = np.random.exponential()
    b = np.arctan(tmp) / alpha
    s = (1.0 + tmp * tmp) ** (1.0 / (2.0 * alpha))
    c1 = alpha * np.sin(v + b) / (np.cos(v) ** (1.0 / alpha))
    c2 = (np.cos(v - alpha * (v + b)) / w) ** ((1.0 - alpha) / alpha)
    return s * c1 * c2


@nb.njit
def _sample_standard_alpha_one(beta: float, seed: Optional[int] = None) -> float:
    """生成标准 α-stable 分布的一个样本（α = 1）"""
    if seed is not None:
        np.random.seed(seed)
    half_pi = np.pi / 2.0
    v = np.random.uniform(-half_pi, half_pi)
    w = np.random.exponential()
    c1 = (half_pi + beta * v) * np.tan(v)
    c2 = np.log((half_pi * w * np.cos(v)) / (half_pi + beta * v)) * beta
    return 2.0 * (c1 - c2) / np.pi


@nb.njit
def sample_stable(
    alpha: float,
    beta: float,
    sigma: float,
    mu: float,
    seed: Optional[int] = None,
) -> float:
    """生成一个 Lévy stable 分布的样本

    参数:
        alpha: 稳定性指数，取值范围 (0, 2]
        beta: 偏度参数，取值范围 [-1, 1]
        sigma: 尺度参数，取值范围 (0, ∞)
        mu: 位置参数，取值范围 (-∞, ∞)
        seed: 随机数种子

    返回:
        一个来自指定参数的 stable 分布的样本
    """
    if not (0 < alpha <= 2):
        raise ValueError("alpha 必须在区间 (0, 2] 内")
    if not (-1 <= beta <= 1):
        raise ValueError("beta 必须在区间 [-1, 1] 内")
    if sigma <= 0:
        raise ValueError("sigma 必须为正数")

    if alpha != 1.0:
        r = _sample_standard_alpha(alpha, beta, seed)
        return sigma * r + mu
    else:
        r = _sample_standard_alpha_one(beta, seed)
        return sigma * r + mu + 2.0 * beta * sigma * np.log(sigma) / np.pi


@nb.njit
def sample_stables(
    alpha: float,
    beta: float,
    sigma: float,
    mu: float,
    size: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """生成多个 Lévy stable 分布的样本

    参数:
        alpha: 稳定性指数，取值范围 (0, 2]
        beta: 偏度参数，取值范围 [-1, 1]
        sigma: 尺度参数，取值范围 (0, ∞)
        mu: 位置参数，取值范围 (-∞, ∞)
        size: 样本数量
        seed: 随机数种子

    返回:
        一个包含指定数量样本的 numpy 数组
    """
    if seed is not None:
        np.random.seed(seed)
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        if alpha != 1.0:
            r = _sample_standard_alpha(alpha, beta)
            result[i] = sigma * r + mu
        else:
            r = _sample_standard_alpha_one(beta)
            result[i] = sigma * r + mu + 2.0 * beta * sigma * np.log(sigma) / np.pi
    return result


@nb.njit
def sample_skew_standard_stable(alpha: float, seed: Optional[int] = None) -> float:
    """生成一个偏斜标准 stable 分布的样本

    参数:
        alpha: 稳定性指数，取值范围 (0, 1)
        seed: 随机数种子

    返回:
        一个来自偏斜标准 stable 分布的样本
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha 必须在区间 (0, 1) 内")
    return _sample_standard_alpha(alpha, 1.0, seed)


@nb.njit
def sample_skew_standard_stables(
    alpha: float, size: int, seed: Optional[int] = None
) -> np.ndarray:
    """生成多个偏斜标准 stable 分布的样本

    参数:
        alpha: 稳定性指数，取值范围 (0, 1)
        size: 样本数量
        seed: 随机数种子

    返回:
        一个包含指定数量样本的 numpy 数组
    """
    if seed is not None:
        np.random.seed(seed)
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = _sample_standard_alpha(alpha, 1.0)
    return result
