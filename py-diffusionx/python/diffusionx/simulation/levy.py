from diffusionx import _core
from typing import Union
from .basic import StochasticProcess, Trajectory
from .utils import check_transform
import numpy as np


real = Union[float, int]


class Levy(StochasticProcess):
    def __init__(
        self,
        alpha: real,
        start_position: real = 0.0,
    ):
        """
        Initialize a Lévy process object.

        Args:
            alpha (real): The alpha parameter of the Lévy process, the value must be in the range (0, 2].
            start_position (real, optional): Starting position of the Lévy process. Defaults to 0.0.

        Raises:
            ValueError: If duration is not positive.
            ValueError: If diffusion coefficient is not positive.
            ValueError: If value is not a number.

        Returns:
            Bm: A Brownian motion object.
        """
        start_position = check_transform(start_position)
        alpha = check_transform(alpha)
        if alpha <= 0 or alpha > 2:
            raise ValueError("alpha must be in the range (0, 2]")

        self.start_position = start_position
        self.alpha = alpha

    def __call__(self, duration: real) -> Trajectory:
        return Trajectory(self, duration)

    def simulate(
        self, duration: real, step_size: real = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the Lévy process.

        Args:
            duration (real): The duration of the Lévy process.
            step_size (real, optional): Step size of the Lévy process. Defaults to 0.01.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the times and positions of the Lévy process.
        """
        step_size = check_transform(step_size)
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        return _core.levy_simulate(
            self.start_position,
            self.alpha,
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
        return _core.levy_fpt(
            self.start_position,
            self.alpha,
            step_size,
            (a, b),
            max_duration,
        )

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        step_size: real = 0.01,
    ):
        """
        Calculate the occupation time of the Lévy process.

        Args:
            domain (tuple[real, real]): The domain of the Lévy process.
            duration (real): The duration of the Lévy process.
            step_size (real, optional): Step size of the Lévy process. Defaults to 0.01.

        Returns:
            real: The occupation time of the Lévy process.
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
        return _core.levy_occupation_time(
            self.start_position,
            self.alpha,
            step_size,
            (a, b),
            duration,
        )


class Subordinator(StochasticProcess):
    def __init__(
        self,
        alpha: real,
    ):
        """
        Initialize a subordinator object.

        Args:
            alpha (real): The alpha parameter of the subordinator, the value must be in the range (0, 1).

        Raises:
            ValueError: If alpha is not in the range (0, 1).

        Returns:
            Subordinator: A subordinator object.
        """
        alpha = check_transform(alpha)
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must be in the range (0, 2]")

        self.alpha = alpha

    def __call__(self, duration: real) -> Trajectory:
        return Trajectory(self, duration)

    def simulate(
        self, duration: real, step_size: real = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the subordinator.

        Args:
            duration (real): The duration of the subordinator.
            step_size (real, optional): Step size of the subordinator. Defaults to 0.01.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the times and positions of the subordinator.
        """
        step_size = check_transform(step_size)
        duration = check_transform(duration)
        if duration <= 0:
            raise ValueError("duration must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        return _core.subordinator_simulate(
            self.alpha,
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
        Calculate the first passage time of the subordinator.

        Args:
            domain (tuple[real, real]): The domain of the subordinator.
            step_size (real, optional): Step size of the subordinator. Defaults to 0.01.

        Returns:
            real: The first passage time of the subordinator.
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
        return _core.subordinator_fpt(
            self.alpha,
            (a, b),
            max_duration,
            step_size,
        )

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        step_size: real = 0.01,
    ):
        """
        Calculate the occupation time of the Lévy process.

        Args:
            domain (tuple[real, real]): The domain of the subordinator.
            duration (real): The duration of the subordinator.
            step_size (real, optional): Step size of the subordinator. Defaults to 0.01.

        Returns:
            real: The occupation time of the subordinator.
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
        return _core.subordinator_occupation_time(
            self.alpha,
            (a, b),
            duration,
            step_size,
        )


class InvSubordinator(StochasticProcess):
    def __init__(
        self,
        alpha: real,
    ):
        """
        Initialize an inverse subordinator object.

        Args:
            alpha (real): The alpha parameter of the inverse subordinator, the value must be in the range (0, 1).

        Raises:
            ValueError: If alpha is not in the range (0, 1).

        Returns:
            InvSubordinator: An inverse subordinator object.
        """
        alpha = check_transform(alpha)
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must be in the range (0, 1)")

        self.alpha = alpha

    def __call__(self, duration: real) -> Trajectory:
        return Trajectory(self, duration)

    def simulate(
        self, duration: real, step_size: real = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        return _core.inv_subordinator_simulate(self.alpha, duration, step_size)

    def fpt(
        self,
        domain: tuple[real, real],
        step_size: real = 0.01,
        max_duration: real = 1000,
    ):
        return _core.inv_subordinator_fpt(
            self.alpha,
            domain,
            max_duration,
            step_size,
        )

    def occupation_time(
        self,
        domain: tuple[real, real],
        duration: real,
        step_size: real = 0.01,
    ):
        return _core.inv_subordinator_occupation_time(
            self.alpha,
            domain,
            duration,
            step_size,
        )
