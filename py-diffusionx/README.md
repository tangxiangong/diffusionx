# DiffusionX

> [!NOTE]
> DiffusionX is a multi-threaded high-performance Rust library for random number/stochastic process simulation, with Python bindings provided via [PyO3](https://github.com/PyO3/pyo3). 

## Usage

```python
from diffusionx.simulation import Bm

# Brownian motion simulation
bm = Bm()
traj = bm(10)
times, positions = traj.simulate(step_size=0.01)  # Simulate Brownian motion trajectory, returns ndarray

# Monte Carlo simulation of Brownian motion statistics
raw_moment = traj.raw_moment(order=1, particles=1000)  # First-order raw moment
central_moment = traj.central_moment(order=2, particles=1000)  # Second-order central moment

# First passage time of Brownian motion
fpt = bm.fpt((-1, 1))
```

## Progress
### Random Number Generation

- [x] Normal distribution
- [x] Uniform distribution
- [x] Exponential distribution
- [x] Poisson distribution
- [x] Alpha-stable distribution

### Stochastic Processes

- [x] Brownian motion
- [ ] Alpha-stable Lévy process
- [ ] Fractional Brownian motion
- [ ] Poisson process
- [ ] Compound Poisson process
- [ ] Langevin equation

## Benchmark

### Test Results

Generating random array of length `10_000_000`

|               | Standard Normal | Uniform [0, 1] |   Stable   |
| :-----------: | :-------------: | :------------: | :--------: |
|  DiffusionX   |     24.1 ms     |   21.687 ms    |  277.6 ms  |
|     Julia     |    28.748 ms    |    9.748 ms    | 713.955 ms |
| NumPy / SciPy |     295 ms      |    81.2 ms     |   3.39 s   |
|     Numba     |        -        |       -        |   1.52 s   |

### Test Environment

#### Hardware Configuration
- Device Model: MacBook Pro 13-inch (2020)
- Processor: Intel Core i5-1038NG7 @ 2.0GHz (4 cores 8 threads)
- Memory: 16GB LPDDR4X 3733MHz

#### Software Environment
- Operating System: macOS Sequoia 15.3
- Rust: 1.85.0-beta.7
- Python: 3.12
- Julia: 1.11
- NumPy: 2.2.2
- SciPy: 1.15.1

## Tech Stack & Features

- 🦀 Rust 2024 Edition
- 🔄 PyO3: Rust/Python bindings
- 🔢 NumPy: Zero-cost array conversion
- 🚀 High performance
- 🔄 Zero-cost NumPy compatibility: All random number generation functions return NumPy arrays directly

## License

This project is dual-licensed under:

* [MIT License](https://opensource.org/licenses/MIT)
* [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

You can choose to use either license. 