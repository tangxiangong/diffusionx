from .basic import StochasticProcess
from .bm import Bm
from .levy import Levy
from typing import Union
from .utils import check_transform

real = Union[int, float]


class FPT:
    def __init__(
        self,
        sp: StochasticProcess,
        domain: tuple[real, real],
        max_duration: real = 1000,
        step_size: real = 0.01,
    ):
        if not isinstance(sp, StochasticProcess):
            raise ValueError("sp must be a StochasticProcess")
        if not isinstance(domain, tuple) or len(domain) != 2:
            raise ValueError("domain must be a tuple of two real numbers")
        a = check_transform(domain[0])
        b = check_transform(domain[1])
        self.max_duration = check_transform(max_duration)
        self.step_size = check_transform(step_size)
        self.sp = sp
        self.domain = (a, b)

    def simulate(self) -> float:
        match self.sp:
            case Bm():
                return self.sp.fpt(self.domain, self.step_size, self.max_duration)
            case Levy():
                return self.sp.fpt(self.domain, self.step_size, self.max_duration)
            case _:
                raise ValueError("sp must be a Brownian motion or a Lévy process")


class OccupationTime:
    def __init__(
        self,
        sp: StochasticProcess,
        domain: tuple[real, real],
        duration: real,
        step_size: real = 0.01,
    ):
        if not isinstance(sp, StochasticProcess):
            raise ValueError("sp must be a StochasticProcess")
        if not isinstance(domain, tuple) or len(domain) != 2:
            raise ValueError("domain must be a tuple of two real numbers")
        a = check_transform(domain[0])
        b = check_transform(domain[1])
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration must be positive")
        else:
            self.duration = duration
        self.step_size = check_transform(step_size)
        self.sp = sp
        self.domain = (a, b)

    def simulate(self) -> float:
        match self.sp:
            case Bm():
                return self.sp.occupation_time(
                    self.domain, self.duration, self.step_size
                )
            case Levy():
                return self.sp.occupation_time(
                    self.domain, self.duration, self.step_size
                )
            case _:
                raise ValueError("sp must be a Brownian motion or a Lévy process")
