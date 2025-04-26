# DiffusionX

English | [简体中文](README-zh.md)
> DiffusionX is a multi-threaded high-performance Rust library for random number generation and stochastic process simulation, designed for scientific computing and quantitative finance applications.

[![docs.rs](https://img.shields.io/badge/docs.rs-latest-blue.svg)](https://docs.rs/diffusionx/latest/diffusionx/)
[![crates.io](https://img.shields.io/crates/v/diffusionx.svg)](https://crates.io/crates/diffusionx)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

## Features

- **High Performance**: Optimized for computational efficiency with multi-threading support via Rayon
- **Comprehensive**: Extensive collection of random distributions and stochastic processes for scientific computing
- **Extensible**: Trait-based architecture enabling easy extension with custom processes and distributions
- **Well-documented**: Detailed API documentation with mathematical background and usage examples
- **Type-safe**: Leverages Rust's type system for compile-time safety and correctness
- **Zero-cost abstractions**: Efficient abstractions with minimal runtime overhead

## Visualization

DiffusionX provides built-in visualization capabilities using the [plotters](https://crates.io/crates/plotters) crate:

- **Process Trajectories**: Easily visualize continuous process trajectories
- **Customizable Plots**: Configure plot appearance including colors, dimensions, and line styles
- **Multiple Backends**: Support for both BitMap and SVG output formats
- **Simple API**: Intuitive trait-based API for visualizing simulation results

## Implemented

### Random Number Generation

- [x] Normal distribution - Gaussian random variables with specified mean and variance
- [x] Uniform distribution - Uniform random variables in specified ranges
- [x] Exponential distribution - Exponential waiting times with specified rate
- [x] Poisson distribution - Discrete count distribution with specified mean
- [x] Alpha-stable distribution - Heavy-tailed distributions with specified stability, skewness, scale, and location

### Stochastic Processes

- [x] Brownian motion - Standard and generalized with drift and diffusion
- [x] Alpha-stable Lévy process - Non-Gaussian processes with heavy tails
- [x] Subordinator - Time-changed processes
- [x] Inverse subordinator - Processes for modeling waiting times
- [x] Poisson process - Counting processes with independent increments
- [x] Fractional Brownian motion - Long-range dependent processes
- [x] Continuous time random walk - Jump processes with random waiting times
- [x] Ornstein-Uhlenbeck process - Mean-reverting processes
- [x] Langevin equation - Physical models with friction and noise
- [x] Generalized Langevin equation - Extended models with memory effects
- [x] Subordinated Langevin equation - Time-changed Langevin processes
- [x] Lévy walk - Superdiffusive processes with coupled jump lengths and waiting times
- [x] Birth-death process - Discrete-state processes with birth and death rates
- [x] Random walk - Discrete-time random walk
- [x] Brownian bridge - Brownian motion conditioned to hit origin at the end
- [x] Brownian excursion - Brownian motion conditioned to be positive and to take the value 0 at time 1
- [x] Brownian meander

## Installation

Add the following to your `Cargo.toml`:
```toml
[dependencies]
diffusionx = "*"  # Replace with the latest version
```

Or use the following command to install:
```bash
cargo add diffusionx
```

## Usage

### Random Number Generation

```rust
use diffusionx::random::{normal, uniform, stable};

// Normal Distribution
let normal_sample = normal::rand(0.0, 1.0)?; // Generate a normal random number with mean 0.0 and std 1.0
let std_normal_samples = normal::standard_rands(1000); // Generate 1000 standard normal random numbers

// Uniform Distribution
let uniform_sample = uniform::range_rand(0..10)?; // Generate a uniform random number in range [0, 10)
let std_uniform_samples = uniform::standard_rands(1000); // Generate 1000 uniform random numbers in range [0, 1)

// α-Stable Distribution
// Standard α-stable distribution (σ=1, μ=0)
let stable_samples = stable::standard_rands(1.5, 0.5, 1000)?; // Generate 1000 standard stable random numbers
```

### Stochastic Process Simulation

```rust
use diffusionx::simulation::{prelude::*, continuous::Bm};

// Brownian motion simulation
let bm = Bm::default();  // Create standard Brownian motion object
let traj = bm.duration(1.0)?;  // Create trajectory with duration 1.0
let (times, positions) = traj.simulate(0.01)?;  // Simulate Brownian motion trajectory with time step 0.01

// Monte Carlo simulation of Brownian motion statistics
let mean = traj.raw_moment(1, 1000, 0.01)?;  // First-order raw moment with 1000 particles
let msd = traj.central_moment(2, 1000, 0.01)?;  // Second-order central moment with 1000 particles

// First passage time of Brownian motion
let fpt = bm.fpt(0.01, (-1.0, 1.0), 1000)?; // Calculate FPT with boundaries at -1.0 and 1.0
```

### Visualization Example

```rust
use diffusionx::{
    simulation::{continuous::Bm, prelude::*},
};

// Create Brownian motion trajectory
let bm = Bm::default();
let traj = bm.duration(10.0)?;

// Configure and create visualization
let config = PlotConfigBuilder::default()
    .time_step(0.01)
    .output_path("brownian_motion.png")
    .caption("Brownian Motion Trajectory")
    .x_label("t")
    .y_label("B")
    .legend("bm")
    .size((800, 600))
    .backend(PlotterBackend::BitMap)
    .build()?;

// Generate plot
traj.plot(&config)?;
```

## Architecture and Extensibility

DiffusionX is designed with a trait-based system for high extensibility and performance:

### Core Traits

- `ContinuousProcess`: Base trait for continuous stochastic processes
- `PointProcess`: Base trait for point processes
- `DiscreteProcess`: Base trait for discrete stochastic processes
- `Moment`: Trait for statistical moments calculation, including raw and central moments
- `Visualize`: Trait for plotting process trajectories

### Functional Distribution Simulation

DiffusionX provides powerful functional distribution simulation for stochastic processes:

1. **First Passage Time (FPT)**: Calculate when a process first reaches a specified boundary
   ```rust
   // For a Brownian motion process
   let bm = Bm::default();
   let fpt = bm.fpt(0.01, (-1.0, 1.0), 1000)?; // Calculate FPT for crossing boundaries
   ```

2. **Occupation Time**: Measure time spent by a process in a specified region
   ```rust
   // For a Brownian motion process
   let bm = Bm::default();
   let traj = bm.duration(10.0)?;
   let occupation = traj.occupation_time(0.01, (0.0, 2.0))?; // Calculate time spent in region [0.0, 2.0]
   ```

### Extending with Custom Processes

1. Adding a New Continuous Process:
   ```rust
   #[derive(Clone)]
   struct MyProcess {
       // Your parameters
       // Should be `Send + Sync` for parallel computation
   }

   impl ContinuousProcess for MyProcess {
       fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<(Vec<f64>, Vec<f64>)> {
           // Implement your simulation logic
           todo!()
       }
   }
   ```

2. Automatic Feature Acquisition:
   - Implementing `ContinuousProcess` trait automatically provides `ContinuousTrajectoryTrait` functionality
   - `ContinuousTrajectory` provides access to the `Moment` trait functionality
   - Built-in support for moment statistics calculation

Example:
```rust
let myprocess = MyProcess::default();
let traj = myprocess.duration(10)?;
let mean = traj.raw_moment(1, 1000, 0.01)?; // Calculate mean with 1000 particles
```

3. Parallel Computing Support:
   - Automatic parallel computation for moment calculations using Rayon
   - Default parallel strategy for statistical calculations
   - Configurable parallelism for optimal performance

4. Visualization Support:
   - Easy trajectory visualization with minimal code
   - Highly customizable plot configuration

Example:
```rust
// Visualize a Brownian motion trajectory
use diffusionx::visualize::{PlotConfigBuilder, Visualize};

let bm = Bm::default().duration(10)?;
let config = PlotConfigBuilder::default()
    .title("Brownian Motion")
    .output_path("brownian_motion.png")
    .build()?;

bm.plot(&config)?; // Generates a plot with the specified configuration
```

## Benchmark
The related content can be found in the **Benchmark** section of [py-diffusionx](https://github.com/tangxiangong/py-diffusionx).

## License

This project is dual-licensed under:

* [MIT License](https://opensource.org/licenses/MIT)
* [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

You can choose to use either license.
