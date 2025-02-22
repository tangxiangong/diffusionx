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
