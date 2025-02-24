# DiffusionX

> Development is in progress. DiffusionX is a multi-threaded high-performance Rust library for random number/stochastic process simulation.

## Usage

### Getting Started
Add the following to your `Cargo.toml`:
```toml
[dependencies]
diffusionx = "*"
```
Or use the following command to install:
```bash
cargo add diffusionx
```

### Random Number Generation

```rust
use diffusionx::random::{normal, uniform, exponential, poisson, stable};

// Normal Distribution
let normal_sample = normal::rand(0.0, 1.0)?; // Generate a normal random number with mean 0.0 and std 1.0
let normal_samples = normal::rands(2.0, 3.0, 1000)?; // Generate 1000 normal random numbers with mean 2.0 and std 3.0
let std_normal_sample = normal::standard_rand(); // Generate a standard normal random number (mean 0, std 1)
let std_normal_samples = normal::standard_rands(1000); // Generate 1000 standard normal random numbers

// Uniform Distribution
let uniform_sample = uniform::range_rand(0..10)?; // Generate a uniform random number in range [0, 10)
let uniform_samples = uniform::range_rands(0..10, 1000)?; // Generate 1000 uniform random numbers in range [0, 10)
let uniform_incl_sample = uniform::inclusive_range_rand(0..=10)?; // Generate a uniform random number in range [0, 10]
let uniform_incl_samples = uniform::inclusive_range_rands(0..=10, 1000)?; // Generate 1000 uniform random numbers in range [0, 10]
let std_uniform_sample = uniform::standard_rand(); // Generate a uniform random number in range [0, 1)
let std_uniform_samples = uniform::standard_rands(1000); // Generate 1000 uniform random numbers in range [0, 1)
let bool_sample = uniform::bool_rand(0.7)?; // Generate a boolean with probability 0.7
let bool_samples = uniform::bool_rands(0.7, 1000)?; // Generate 1000 booleans with probability 0.7

// Exponential Distribution
let exp_sample = exponential::rand(1.0)?; // Generate an exponential random number with rate 1.0
let exp_samples = exponential::rands(1.0, 1000)?; // Generate 1000 exponential random numbers with rate 1.0

// Poisson Distribution
let poisson_sample = poisson::rand(5.0)?; // Generate a Poisson random number with mean 5.0
let poisson_samples = poisson::rands(5.0, 1000)?; // Generate 1000 Poisson random numbers with mean 5.0

// α-Stable Distribution
// Standard α-stable distribution (σ=1, μ=0)
let stable_sample = stable::standard_rand(1.5, 0.5)?; // Generate a standard stable random number with α=1.5, β=0.5
let stable_samples = stable::standard_rands(1.5, 0.5, 1000)?; // Generate 1000 standard stable random numbers

// General α-stable distribution
let stable_sample = stable::rand(1.5, 0.5, 1.0, 0.0)?; // Generate a stable random   number with α=1.5, β=0.5, σ=1.0, μ=0.0
let stable_samples = stable::rands(1.5, 0.5, 1.0, 0.0, 1000)?; // Generate 1000 stable random numbers

// Special cases of α-stable distribution
let skew_sample = stable::skew_rand(1.5)?; // Generate a totally skewed stable random number with α=1.5
let skew_samples = stable::skew_rands(1.5, 1000)?; // Generate 1000 totally skewed stable random numbers
let sym_sample = stable::sym_standard_rand(1.5)?; // Generate a symmetric stable random number with α=1.5
let sym_samples = stable::sym_standard_rands(1.5, 1000)?; // Generate 1000 symmetric stable random numbers

// Object-oriented interface for stable distributions
let stable = stable::Stable::new(1.5, 0.5, 1.0, 0.0)?; // Create a stable distribution object
let samples = stable.samples(1000)?; // Generate 1000 samples

let std_stable = stable::StandardStable::new(1.5, 0.5)?; // Create a standard stable distribution object
let samples = std_stable.samples(1000)?; // Generate 1000 samples
```

### Stochastic Process Simulation

```rust
use diffusionx::simulation::{prelude::*, Bm};

// Brownian motion simulation
let bm = Bm::default();  // Create standard Brownian motion object
let traj = bm.duration(1.0)?;  // Create trajectory with duration 1.0
let (times, positions) = traj.simulate(0.01)?;  // Simulate Brownian motion trajectory with time step 0.01

// Monte Carlo simulation of Brownian motion statistics
let mean = traj.raw_moment(1, 1000, 0.01)?;  // First-order raw moment with 1000 particles
let msd = traj.central_moment(2, 1000, 0.01)?;  // Second-order central moment with 1000 particles

// First passage time of Brownian motion
let max_duration = 1000; // if over this duration, the simulation will be terminated and return None
let fpt = bm.fpt(0.01, (-1.0, 1.0), max_duration)?; 
// or
let fpt = FirstPassageTime::new(&bm, (-1.0, 1.0))?;
let fpt_result = fpt.simulate(max_duration, 0.01)?;
```
## Extensibility

DiffusionX is designed with a trait-based system for high extensibility:

### Core Traits

- `ContinuousProcess`: Base trait for continuous stochastic processes
- `PointProcess`: Base trait for point processes
- `Moment`: Trait for statistical moments calculation, including raw and central moments

### Feature Extension

1. Adding New Continuous Process:
   ```rust
   #[derive(Clone)]
   struct MyProcess {
       // Your parameters
       // Should be `Send + Sync`
   }
   
   impl ContinuousProcess for MyProcess {
       fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<(Vec<f64>, Vec<f64>)> {
           // Implement your simulation logic
           todo!()
       }
   }
   ```

2. Automatic Feature Acquisition:
   - Get `ContinuousTrajectoryTrait` functionality automatically by implementing `ContinuousProcess` trait
   - Get `Moment` trait functionality through `ContinuousTrajectory`
   - Built-in support for moment statistics calculation

Example:
```rust
let myprocess = MyProcess::default();
let traj = myprocess.duration(10)?;
let (times, positions) = traj.simulate(0.01)?;
let mean = traj.raw_moment(1, 1000, 0.01)?;
let msd = traj.central_moment(2, 1000, 0.01)?;
```

3. Parallel Computing Support:
   - Automatic parallel computation support for moment calculations
   - Default parallel strategy for statistical calculations


## Progress
### Random Number Generation

- [x] Normal distribution
- [x] Uniform distribution
- [x] Exponential distribution
- [x] Poisson distribution
- [x] Alpha-stable distribution

### Stochastic Processes

- [x] Brownian motion
- [x] Alpha-stable Lévy process
- [x] Subordinator
- [x] Inverse subordinator
- [x] Poisson process
- [ ] Fractional Brownian motion
- [ ] Compound Poisson process
- [ ] Langevin equation

## Benchmark

### Test Results

Generating random array of length `10_000_000`

|               | Standard Normal | Uniform [0, 1] |   Stable   |
| :-----------: | :-------------: | :------------: | :--------: |
|  DiffusionX   |    17.576 ms   |     15.131 ms     | 133.85 ms  |
|     Julia     |    27.671 ms   |     12.755 ms      | 570.260 ms |
| NumPy / SciPy |     199 ms    |      66.6 ms      |   1.67 s   |
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

## License

This project is dual-licensed under:

* [MIT License](https://opensource.org/licenses/MIT)
* [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

You can choose to use either license. 