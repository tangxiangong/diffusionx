from diffusionx import _core
from typing import Union
from .basic import StochasticProcess, Trajectory
from .utils import check_transform
import numpy as np


real = Union[float, int]


class Poisson(StochasticProcess):
    def __init__(
        self,
        lambda_: real = 1.0,
    ):
        """
        Initialize a Poisson process object.

        Args:
            lambda_ (real, optional): Lambda of the Poisson process. Defaults to 1.0.

        Raises:
            ValueError: If lambda_ is not positive.
            ValueError: If lambda_ is not a number.

        Returns:
            Poisson: A Poisson process object.
        """
        lambda_ = check_transform(lambda_)
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")
        self.lambda_ = lambda_

    def __call__(self, duration: real) -> Trajectory:
        return Trajectory(self, duration)

    def simulate(
        self, duration: real, step_size: real = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the Poisson process.

        Args:
            duration (real): Duration of the Poisson process.
            step_size (real, optional): Step size of the Poisson process. Defaults to 0.01.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the times and positions of the Poisson process.
        """
        step_size = check_transform(step_size)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration must be positive")
        return _core.poisson_simulate_duration(
            self.lambda_,
            duration,
        )

    def simulate_step(self, num_step: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the Poisson process.

        Args:
            num_step (int): Number of steps of the Poisson process.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the times and positions of the Poisson process.
        """
        if not isinstance(num_step, int):
            raise ValueError("num_step must be an integer")
        if num_step <= 0:
            raise ValueError("num_step must be positive")
        return _core.poisson_simulate_step(
            self.lambda_,
            num_step,
        )

    def raw_moment(self, duration: real, order: int, particles: int) -> float:
        """
        Calculate the raw moment of the Poisson process.

        Args:
            duration (real): Duration of the Poisson process.
            order (int): Order of the moment.
            particles (int): Number of particles.


        Returns:
            real: The raw moment of the Poisson process.
        """
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration must be positive")
        if not isinstance(order, int):
            raise ValueError("order must be an integer")
        if order < 0:
            raise ValueError("order must be non-negative")
        if not isinstance(particles, int):
            raise ValueError("particles must be an integer")
        if particles <= 0:
            raise ValueError("particles must be positive")
        return _core.poisson_raw_moment(
            self.lambda_,
            duration,
            order,
            particles,
        )

    def central_moment(self, duration: real, order: int, particles: int) -> float:
        """
        Calculate the central moment of the Poisson process.

        Args:
            duration (real): Duration of the Poisson process.
            order (int): Order of the moment.
            particles (int): Number of particles.

        Returns:
            real: The raw moment of the Brownian motion.
        """
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration must be positive")
        if not isinstance(order, int):
            raise ValueError("order must be an integer")
        if order < 0:
            raise ValueError("order must be non-negative")
        if not isinstance(particles, int):
            raise ValueError("particles must be an integer")
        if particles <= 0:
            raise ValueError("particles must be positive")
        return _core.poisson_central_moment(
            self.lambda_,
            duration,
            order,
            particles,
        )

    def fpt(
        self,
        domain: tuple[real, real],
        max_duration: real = 1000,
    ):
        """
        Calculate the first passage time of the Poisson process.

        Args:
            domain (tuple[real, real]): The domain of the Poisson process.
            max_duration (real, optional): The maximum duration of the Poisson process. Defaults to 1000.

        Raises:
            ValueError: If domain is not a tuple of two real numbers.
            ValueError: If max_duration is not a real number.
            ValueError: If max_duration is not positive.

        Returns:
            real: The first passage time of the Poisson process.
        """
        if not isinstance(domain, tuple):
            raise ValueError("domain must be a tuple")
        if len(domain) != 2:
            raise ValueError("domain must be a tuple of two real numbers")
        max_duration = check_transform(max_duration)
        if max_duration <= 0:
            raise ValueError("max_duration must be positive")
        return _core.poisson_fpt(
            self.lambda_,
            domain,
            max_duration,
        )

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
    ):
        """
        Calculate the occupation time of the Brownian motion.

        Args:
            domain (tuple[real, real]): The domain of the Brownian motion.
            duration (real): The duration of the Brownian motion.

        Returns:
            real: The occupation time of the Brownian motion.
        """
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration must be positive")
        a = check_transform(domain[0])
        b = check_transform(domain[1])
        if a >= b:
            raise ValueError("domain must be a valid interval")
        return _core.poisson_occupation_time(
            self.lambda_,
            domain,
            duration,
        )
