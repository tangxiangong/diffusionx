from diffusionx import _core
from typing import Union
from .basic import StochasticProcess, Trajectory
from .utils import check_transform
import numpy as np


real = Union[float, int]


class Bm(StochasticProcess):
    def __init__(
        self,
        start_position: real = 0.0,
        diffusion_coefficient: real = 1.0,
    ):
        """
        Initialize a Brownian motion object.

        Args:
            start_position (real, optional): Starting position of the Brownian motion. Defaults to 0.0.
            diffusion_coefficient (real, optional): Diffusion coefficient of the Brownian motion. Defaults to 1.0.

        Raises:
            ValueError: If duration is not positive.
            ValueError: If diffusion coefficient is not positive.
            ValueError: If value is not a number.

        Returns:
            Bm: A Brownian motion object.
        """
        start_position = check_transform(start_position)
        diffusion_coefficient = check_transform(diffusion_coefficient)
        if diffusion_coefficient <= 0:
            raise ValueError("diffusion_coefficient must be positive")

        self.start_position = start_position
        self.diffusion_coefficient = diffusion_coefficient

    def __call__(self, duration: real) -> Trajectory:
        return Trajectory(self, duration)

    def simulate(
        self, duration: real, step_size: real = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the Brownian motion.

        Args:
            step_size (real, optional): Step size of the Brownian motion. Defaults to 0.01.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the times and positions of the Brownian motion.
        """
        step_size = check_transform(step_size)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration must be positive")
        return _core.bm_simulate(
            self.start_position,
            self.diffusion_coefficient,
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
        Calculate the first passage time of the Brownian motion.

        Args:
            domain (tuple[real, real]): The domain of the Brownian motion.
            step_size (real, optional): Step size of the Brownian motion. Defaults to 0.01.

        Returns:
            real: The first passage time of the Brownian motion.
        """
        step_size = check_transform(step_size)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        a = check_transform(domain[0])
        b = check_transform(domain[1])
        if a >= b:
            raise ValueError("domain must be a valid interval")
        max_duration = check_transform(max_duration)
        if max_duration <= 0:
            raise ValueError("max_duration must be positive")
        return _core.bm_fpt(
            self.start_position,
            self.diffusion_coefficient,
            step_size,
            (a, b),
            max_duration,
        )

    def raw_moment(
        self, duration: real, order: int, particles: int, step_size: float = 0.01
    ) -> float:
        """
        Calculate the raw moment of the Brownian motion.

        Args:
            order (int): Order of the moment.
            particles (int): Number of particles.
            step_size (real, optional): Step size of the Brownian motion. Defaults to 0.01.

        Returns:
            real: The raw moment of the Brownian motion.
        """
        if not isinstance(order, int):
            raise ValueError("order must be an integer")
        elif order < 0:
            raise ValueError("order must be non-negative")
        elif order == 0:
            return 1
        if not isinstance(particles, int):
            raise ValueError("particles must be an integer")
        elif particles <= 0:
            raise ValueError("particles must be positive")
        step_size = check_transform(step_size)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        return _core.bm_raw_moment(
            self.start_position,
            self.diffusion_coefficient,
            duration,
            step_size,
            order,
            particles,
        )

    def central_moment(
        self, duration: real, order: int, particles: int, step_size: float = 0.01
    ):
        """
        Calculate the central moment of the Brownian motion.

        Args:
            order (int): Order of the moment.
            particles (int): Number of particles.
            step_size (float, optional): Step size of the Brownian motion. Defaults to 0.01.

        Raises:
            ValueError: If order is not an integer.
            ValueError: If order is negative.
            ValueError: If order is zero.
            ValueError: If particles is not an integer.
            ValueError: If particles is not positive.
            ValueError: If step_size is not positive.

        Returns:
            real: The central moment of the Brownian motion.
        """
        if not isinstance(order, int):
            raise ValueError("order must be an integer")
        elif order < 0:
            raise ValueError("order must be non-negative")
        elif order == 0:
            return 1
        if not isinstance(particles, int):
            raise ValueError("particles must be an integer")
        elif particles <= 0:
            raise ValueError("particles must be positive")
        step_size = check_transform(step_size)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        return _core.bm_central_moment(
            self.start_position,
            self.diffusion_coefficient,
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
        Calculate the occupation time of the Brownian motion.

        Args:
            domain (tuple[real, real]): The domain of the Brownian motion.
            duration (real): The duration of the Brownian motion.
            step_size (real, optional): Step size of the Brownian motion. Defaults to 0.01.

        Returns:
            real: The occupation time of the Brownian motion.
        """
        step_size = check_transform(step_size)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration must be positive")
        a = check_transform(domain[0])
        b = check_transform(domain[1])
        if a >= b:
            raise ValueError("domain must be a valid interval")
        return _core.bm_occupation_time(
            self.start_position,
            self.diffusion_coefficient,
            step_size,
            (a, b),
            duration,
        )
