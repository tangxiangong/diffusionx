from abc import ABC, abstractmethod
from typing import Union
import numpy as np


real = Union[int, float]


class StochasticProcess(ABC):
    @abstractmethod
    def simulate(
        self, duration: real, step_size: real
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def raw_moment(
        self, duration: real, order: int, particles: int, step_size: float = 0.01
    ) -> float:
        pass

    @abstractmethod
    def central_moment(
        self, duration: real, order: int, particles: int, step_size: float = 0.01
    ) -> float:
        pass

    @abstractmethod
    def fpt(
        self,
        domain: tuple[real, real],
        step_size: real = 0.01,
        max_duration: real = 1000,
    ) -> float:
        pass


class Trajectory:
    def __init__(self, sp: StochasticProcess, duration: real):
        if not isinstance(sp, StochasticProcess):
            raise ValueError("sp must be a StochasticProcess")
        if isinstance(duration, int):
            duration = float(duration)
        elif not isinstance(duration, float):
            raise ValueError("duration must be a real number")
        if duration <= 0:
            raise ValueError("duration must be positive")
        self.sp = sp
        self.duration = duration

    def simulate(self, step_size: real = 0.01) -> tuple[np.ndarray, np.ndarray]:
        return self.sp.simulate(self.duration, step_size)

    def raw_moment(self, duration, order: int, particles: int, step_size: float = 0.01):
        return self.sp.raw_moment(duration, order, particles, step_size)

    def central_moment(
        self, duration: real, order: int, particles: int, step_size: float = 0.01
    ):
        return self.sp.central_moment(duration, order, particles, step_size)
