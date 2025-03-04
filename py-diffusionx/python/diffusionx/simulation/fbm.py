from diffusionx import _core
from typing import Union
from .basic import StochasticProcess, Trajectory
from .utils import check_transform
import numpy as np


real = Union[float, int]


class Fbm(StochasticProcess):
    def __init__(
        self,
        start_position: real = 0.0,
        hurst_exponent: real = 0.5,
    ):
        """
        初始化分数布朗运动对象。

        参数:
            start_position (real, optional): 分数布朗运动的起始位置。默认为 0.0。
            hurst_exponent (real, optional): 分数布朗运动的 Hurst 指数。默认为 0.5。

        异常:
            ValueError: 如果 Hurst 指数不在 (0, 1) 范围内。
            ValueError: 如果值不是数字。

        返回:
            Fbm: 分数布朗运动对象。
        """
        start_position = check_transform(start_position)
        hurst_exponent = check_transform(hurst_exponent)
        if hurst_exponent <= 0 or hurst_exponent >= 1:
            raise ValueError("hurst_exponent 必须在 (0, 1) 范围内")

        self.start_position = start_position
        self.hurst_exponent = hurst_exponent

    def __call__(self, duration: real) -> Trajectory:
        return Trajectory(self, duration)

    def simulate(
        self, duration: real, step_size: real = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        模拟分数布朗运动。

        参数:
            duration (real): 模拟的持续时间。
            step_size (real, optional): 分数布朗运动的步长。默认为 0.01。

        返回:
            tuple[np.ndarray, np.ndarray]: 包含分数布朗运动的时间和位置的元组。
        """
        step_size = check_transform(step_size)
        if step_size <= 0:
            raise ValueError("step_size 必须为正数")
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration 必须为正数")
        return _core.fbm_simulate(
            self.start_position,
            self.hurst_exponent,
            duration,
            step_size,
        )

    def fpt(
        self,
        domain: tuple[real, real],
        step_size: real = 0.01,
        max_duration: real = 1000,
    ):
        """
        计算分数布朗运动的首次通过时间。

        参数:
            domain (tuple[real, real]): 分数布朗运动的区域。
            step_size (real, optional): 分数布朗运动的步长。默认为 0.01。
            max_duration (real, optional): 最大持续时间。默认为 1000。

        返回:
            real: 分数布朗运动的首次通过时间。
        """
        step_size = check_transform(step_size)
        if step_size <= 0:
            raise ValueError("step_size 必须为正数")
        a = check_transform(domain[0])
        b = check_transform(domain[1])
        if a >= b:
            raise ValueError("domain 必须是有效区间")
        max_duration = check_transform(max_duration)
        if max_duration <= 0:
            raise ValueError("max_duration 必须为正数")
        return _core.fbm_fpt(
            self.start_position,
            self.hurst_exponent,
            step_size,
            (a, b),
            max_duration,
        )

    def raw_moment(
        self, duration: real, order: int, particles: int, step_size: float = 0.01
    ) -> float:
        """
        计算分数布朗运动的原始矩。

        参数:
            duration (real): 持续时间。
            order (int): 矩的阶数。
            particles (int): 粒子数量。
            step_size (real, optional): 分数布朗运动的步长。默认为 0.01。

        返回:
            real: 分数布朗运动的原始矩。
        """
        if not isinstance(order, int):
            raise ValueError("order 必须为整数")
        elif order < 0:
            raise ValueError("order 必须为非负数")
        elif order == 0:
            return 1
        if not isinstance(particles, int):
            raise ValueError("particles 必须为整数")
        elif particles <= 0:
            raise ValueError("particles 必须为正数")
        step_size = check_transform(step_size)
        if step_size <= 0:
            raise ValueError("step_size 必须为正数")
        return _core.fbm_raw_moment(
            self.start_position,
            self.hurst_exponent,
            duration,
            step_size,
            order,
            particles,
        )

    def central_moment(
        self, duration: real, order: int, particles: int, step_size: float = 0.01
    ):
        """
        计算分数布朗运动的中心矩。

        参数:
            duration (real): 持续时间。
            order (int): 矩的阶数。
            particles (int): 粒子数量。
            step_size (float, optional): 分数布朗运动的步长。默认为 0.01。

        异常:
            ValueError: 如果阶数不是整数。
            ValueError: 如果阶数为负数。
            ValueError: 如果阶数为零。
            ValueError: 如果粒子数量不是整数。
            ValueError: 如果粒子数量不是正数。
            ValueError: 如果步长不是正数。

        返回:
            real: 分数布朗运动的中心矩。
        """
        if not isinstance(order, int):
            raise ValueError("order 必须为整数")
        elif order < 0:
            raise ValueError("order 必须为非负数")
        elif order == 0:
            return 1
        if not isinstance(particles, int):
            raise ValueError("particles 必须为整数")
        elif particles <= 0:
            raise ValueError("particles 必须为正数")
        step_size = check_transform(step_size)
        if step_size <= 0:
            raise ValueError("step_size 必须为正数")
        return _core.fbm_central_moment(
            self.start_position,
            self.hurst_exponent,
            duration,
            step_size,
            order,
            particles,
        )

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        step_size: real = 0.01,
    ):
        """
        计算分数布朗运动的占据时间。

        参数:
            domain (tuple[real, real]): 分数布朗运动的区域。
            duration (real): 分数布朗运动的持续时间。
            step_size (real, optional): 分数布朗运动的步长。默认为 0.01。

        返回:
            real: 分数布朗运动的占据时间。
        """
        step_size = check_transform(step_size)
        if step_size <= 0:
            raise ValueError("step_size 必须为正数")
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration 必须为正数")
        a = check_transform(domain[0])
        b = check_transform(domain[1])
        if a >= b:
            raise ValueError("domain 必须是有效区间")
        return _core.fbm_occupation_time(
            self.start_position,
            self.hurst_exponent,
            step_size,
            (a, b),
            duration,
        )

    def mean(self, duration: real, particles: int, step_size: float = 0.01) -> float:
        """
        计算分数布朗运动的均值。

        参数:
            duration (real): 持续时间。
            particles (int): 粒子数量。
            step_size (float, optional): 分数布朗运动的步长。默认为 0.01。

        返回:
            float: 分数布朗运动的均值。
        """
        return self.raw_moment(duration, 1, particles, step_size)

    def msd(self, duration: real, particles: int, step_size: float = 0.01) -> float:
        """
        计算分数布朗运动的均方位移。

        参数:
            duration (real): 持续时间。
            particles (int): 粒子数量。
            step_size (float, optional): 分数布朗运动的步长。默认为 0.01。

        返回:
            float: 分数布朗运动的均方位移。
        """
        return self.central_moment(duration, 2, particles, step_size)
