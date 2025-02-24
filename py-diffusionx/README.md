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
|  DiffusionX   |     41.2 ms     |   34.3 ms    |  293 ms  |
|     Julia     |    27.671 ms    |    12.755 ms    | 570.260 ms |
| NumPy / SciPy |     199 ms      |    66.6 ms     |   1.67 s   |
|     Numba     |        -        |       -        |   1.15 s   |

### Test Environment

#### Hardware Configuration
- Device Model: MacBook Air 13-inch (2024)
- Processor: Apple M3 
- Memory: 16GB

#### Software Environment
- Operating System: macOS Sequoia 15.3
- Rust: 1.85.0
- Python: 3.12
- Julia: 1.11
- NumPy: 2
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