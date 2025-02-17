# DiffusionX

English | [简体中文](README-zh.md)

> [!NOTE]
> Development is in progress. DiffusionX is a multi-threaded high-performance Rust library for random number/stochastic process simulation, with Python bindings provided via [PyO3](https://github.com/PyO3/pyo3).

## Usage
### Python

```python
from diffusionx.simulation import Bm

# Brownian motion simulation
bm = Bm(10) 
times, positions = bm.simulate(step_size=0.01)  # Simulate Brownian motion trajectory, returns ndarray

# Monte Carlo simulation of Brownian motion statistics
raw_moment = bm.raw_moment(order=1, particles=1000)  # First-order raw moment
central_moment = bm.central_moment(order=2, particles=1000)  # Second-order central moment

# First passage time of Brownian motion
fpt = bm.fpt((-1, 1))
```

### Rust

```rust
use diffusionx::simulation::{Bm, Simulation, Functional};

// Brownian motion simulation
let bm = Bm::new(0.0, 1.0, 1.0)?;  // Create Brownian motion object: initial position 0, diffusion coefficient 1, duration 1
let time_step = 0.01;  // Time step
let (times, positions) = bm.simulate(time_step)?;  // Simulate Brownian motion trajectory

// Monte Carlo simulation of Brownian motion statistics
let mean = bm.mean(time_step, 1000)?;  // Mean  bm.raw_moment(time_step, 1, 1000)?;
let msd = bm.msd(time_step, 1000)?;  // Mean square displacement  bm.central_moment(time_step, 2, 1000)?;

// First passage time of Brownian motion
let max_duration = 1000; // if over this duration, the simulation will be terminated and return None
let fpt = bm.fpt(time_step, (-1.0, 1.0), max_duration)?; 
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

|                          | Standard Normal | Uniform [0, 1] |   Stable   |
| :----------------------: | :-------------: | :------------: | :--------: |
|  DiffusionX (Rust ver.)  |    23.811 ms    |   20.450 ms    | 273.68 ms  |
| DiffusionX (Python ver.) |     24.1 ms     |   21.687 ms    |  277.6 ms  |
|          Julia           |    28.748 ms    |    9.748 ms    | 713.955 ms |
|      NumPy / SciPy       |     295 ms      |    81.2 ms     |   3.39 s   |
|          Numba           |        -        |       -        |   1.52 s   |

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
- 🚀 High Performance
- 🔄 Zero-cost NumPy compatibility: All random number generation functions directly return NumPy arrays, no extra conversion needed

## License

This project is dual-licensed under:

* [MIT License](https://opensource.org/licenses/MIT)
* [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

You can choose to use either license. 