from diffusionx import _core
from typing import Union
from .basic import StochasticProcess, Trajectory
from .utils import check_transform
import numpy as np


real = Union[float, int]


class CTRW(StochasticProcess):
    def __init__(
        self,
        alpha: real = 1.0,
        beta: real = 2.0,
        start_position: real = 0.0,
    ):
        """
        初始化连续时间随机游走对象。

        参数:
            alpha (real, optional): 等待时间分布的指数，在 0 到 1 之间。当 alpha = 1 时，等待时间为指数分布，否则为幂律分布，尾指数为 alpha。默认为 1.0。
            beta (real, optional): 跳跃长度分布的指数，在 0 到 2 之间。当 beta = 2 时，跳跃长度为正态分布，否则为幂律分布，尾指数为 beta。默认为 2.0。
            start_position (real, optional): 连续时间随机游走的起始位置。默认为 0.0。

        异常:
            ValueError: 如果 alpha 不在 (0, 1] 范围内。
            ValueError: 如果 beta 不在 (0, 2] 范围内。
            ValueError: 如果值不是数字。

        返回:
            CTRW: 连续时间随机游走对象。
        """
        alpha = check_transform(alpha)
        beta = check_transform(beta)
        start_position = check_transform(start_position)

        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha 必须在 (0, 1] 范围内")
        if beta <= 0 or beta > 2:
            raise ValueError("beta 必须在 (0, 2] 范围内")

        self.alpha = alpha
        self.beta = beta
        self.start_position = start_position

    def __call__(self, duration: real) -> Trajectory:
        return Trajectory(self, duration)

    def simulate(
        self, duration: real, step_size: real = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        模拟连续时间随机游走。

        参数:
            duration (real): 模拟的持续时间。
            step_size (real, optional): 步长，在此模拟中不使用，但为了与其他随机过程接口一致而保留。默认为 0.01。

        返回:
            tuple[np.ndarray, np.ndarray]: 包含连续时间随机游走的时间和位置的元组。
        """
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration 必须为正数")
        return _core.ctrw_simulate_duration(
            self.alpha,
            self.beta,
            self.start_position,
            duration,
        )

    def simulate_with_step(self, num_step: int) -> tuple[np.ndarray, np.ndarray]:
        """
        使用指定步数模拟连续时间随机游走。

        参数:
            num_step (int): 模拟的步数。

        返回:
            tuple[np.ndarray, np.ndarray]: 包含连续时间随机游走的时间和位置的元组。
        """
        if not isinstance(num_step, int):
            raise ValueError("num_step 必须为整数")
        if num_step <= 0:
            raise ValueError("num_step 必须为正数")
        return _core.ctrw_simulate_step(
            self.alpha,
            self.beta,
            self.start_position,
            num_step,
        )

    def fpt(
        self,
        domain: tuple[real, real],
        max_duration: real = 1000,
    ):
        """
        计算连续时间随机游走的首次通过时间。

        参数:
            domain (tuple[real, real]): 连续时间随机游走的区域。
            max_duration (real, optional): 最大持续时间。默认为 1000。

        返回:
            real: 连续时间随机游走的首次通过时间。
        """
        a = check_transform(domain[0])
        b = check_transform(domain[1])
        if a >= b:
            raise ValueError("domain 必须是有效区间")
        max_duration = check_transform(max_duration)
        if max_duration <= 0:
            raise ValueError("max_duration 必须为正数")
        return _core.ctrw_fpt(
            self.alpha,
            self.beta,
            self.start_position,
            (a, b),
            max_duration,
        )

    def raw_moment(self, duration: real, order: int, particles: int) -> float:
        """
        计算连续时间随机游走的原始矩。

        参数:
            duration (real): 持续时间。
            order (int): 矩的阶数。
            particles (int): 粒子数量。

        返回:
            real: 连续时间随机游走的原始矩。
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
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration 必须为正数")
        return _core.ctrw_raw_moment(
            self.alpha,
            self.beta,
            self.start_position,
            duration,
            order,
            particles,
        )

    def central_moment(self, duration: real, order: int, particles: int):
        """
        计算连续时间随机游走的中心矩。

        参数:
            duration (real): 持续时间。
            order (int): 矩的阶数。
            particles (int): 粒子数量。

        异常:
            ValueError: 如果阶数不是整数。
            ValueError: 如果阶数为负数。
            ValueError: 如果阶数为零。
            ValueError: 如果粒子数量不是整数。
            ValueError: 如果粒子数量不是正数。

        返回:
            real: 连续时间随机游走的中心矩。
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
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration 必须为正数")
        return _core.ctrw_central_moment(
            self.alpha,
            self.beta,
            self.start_position,
            duration,
            order,
            particles,
        )

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
    ):
        """
        计算连续时间随机游走的占据时间。

        参数:
            domain (tuple[real, real]): 连续时间随机游走的区域。
            duration (real): 连续时间随机游走的持续时间。

        返回:
            real: 连续时间随机游走的占据时间。
        """
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration 必须为正数")
        a = check_transform(domain[0])
        b = check_transform(domain[1])
        if a >= b:
            raise ValueError("domain 必须是有效区间")
        return _core.ctrw_occupation_time(
            self.alpha,
            self.beta,
            self.start_position,
            (a, b),
            duration,
        )

    def mean(self, duration: real, particles: int) -> float:
        """
        计算连续时间随机游走的均值。

        参数:
            duration (real): 持续时间。
            particles (int): 粒子数量。

        返回:
            float: 连续时间随机游走的均值。
        """
        return self.raw_moment(duration, 1, particles)

    def msd(self, duration: real, particles: int) -> float:
        """
        计算连续时间随机游走的均方位移。

        参数:
            duration (real): 持续时间。
            particles (int): 粒子数量。

        返回:
            float: 连续时间随机游走的均方位移。
        """
        return self.central_moment(duration, 2, particles)
