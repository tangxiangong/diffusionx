import numpy as np

def exp_rand(scale: float = 1.0) -> float: ...
def exp_rands(n: int, scale: float = 1.0) -> np.ndarray: ...
def uniform_rand_float(
    low: float = 0.0, high: float = 1.0, end: bool = False
) -> float: ...
def uniform_rands_float(
    n: int, low: float = 0.0, high: float = 1.0, end: bool = False
) -> np.ndarray: ...
def uniform_rand_int(low: int, high: int, end: bool = False) -> int: ...
def uniform_rands_int(n: int, low: int, high: int, end: bool = False) -> np.ndarray: ...
def normal_rand(mu: float = 0.0, sigma: float = 1.0) -> float: ...
def normal_rands(n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray: ...
def poisson_rand(lambda_: float = 1.0) -> int: ...
def poisson_rands(n: int, lambda_: float = 1.0) -> np.ndarray: ...
def stable_rand(
    alpha: float, beta: float, sigma: float = 1.0, mu: float = 0.0
) -> float: ...
def stable_rands(
    n: int, alpha: float, beta: float, sigma: float = 1.0, mu: float = 0.0
) -> np.ndarray: ...
def skew_stable_rand(alpha: float) -> float: ...
def skew_stable_rands(n: int, alpha: float) -> np.ndarray: ...
def bool_rand(p: float = 0.5) -> bool: ...
def bool_rands(n: int, p: float = 0.5) -> np.ndarray: ...
def bm_simulate(
    start_position: float,
    diffusion_coefficient: float,
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def bm_raw_moment(
    start_position: float,
    diffusion_coefficient: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def bm_central_moment(
    start_position: float,
    diffusion_coefficient: float,
    duration: float,
    step_size: float,
    order: int,
    particles: int,
) -> float: ...
def bm_fpt(
    start_position: float,
    diffusion_coefficient: float,
    step_size: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float: ...
def levy_simulate(
    start_position: float,
    alpha: float,
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def levy_fpt(
    start_position: float,
    alpha: float,
    step_size: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float: ...
def bm_occupation_time(
    start_position: float,
    diffusion_coefficient: float,
    step_size: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def levy_occupation_time(
    start_position: float,
    alpha: float,
    step_size: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def poisson_simulate_duration(
    lambda_: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def poisson_simulate_step(
    lambda_: float,
    num_step: int,
) -> tuple[np.ndarray, np.ndarray]: ...
def poisson_raw_moment(
    lambda_: float,
    duration: float,
    order: int,
    particles: int,
) -> float: ...
def poisson_central_moment(
    lambda_: float,
    duration: float,
    order: int,
    particles: int,
) -> float: ...
def poisson_fpt(
    lambda_: float,
    domain: tuple[float, float],
    max_duration: float,
) -> float: ...
def poisson_occupation_time(
    lambda_: float,
    domain: tuple[float, float],
    duration: float,
) -> float: ...
def subordinator_simulate(
    alpha: float,
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def subordinator_fpt(
    alpha: float,
    domain: tuple[float, float],
    max_duration: float,
    step_size: float,
) -> float: ...
def subordinator_occupation_time(
    alpha: float,
    domain: tuple[float, float],
    duration: float,
    step_size: float,
) -> float: ...
def inv_subordinator_simulate(
    alpha: float,
    duration: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]: ...
def inv_subordinator_fpt(
    alpha: float,
    domain: tuple[float, float],
    max_duration: float,
    step_size: float,
) -> float: ...
def inv_subordinator_occupation_time(
    alpha: float,
    domain: tuple[float, float],
    duration: float,
    step_size: float,
) -> float: ...
