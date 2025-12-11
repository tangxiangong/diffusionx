<div align=center>
<h1 aligh="center">
DiffusionX
</h1>
<p align="center">
A multi-threaded high-performance Rust library for random number generation and stochastic process simulation, with optional GPU acceleration.
</p>
<p align="center">
Dedicated to my brief yet unforgettable years in LZU and to XX.
</p>
<p align="center">
English | <a href="README-zh.md">简体中文</a>
</p>
<p align="center">
<a href="https://crates.io/crates/diffusionx"> <img alt="Crates.io Version" src="https://img.shields.io/crates/v/diffusionx?style=for-the-badge"> </a>
<a href="https://docs.rs/diffusionx"> <img alt="docs.rs" src="https://img.shields.io/docsrs/diffusionx?style=for-the-badge"> </a>
<img alt="License: MIT OR Apache-2.0" src="https://img.shields.io/crates/l/diffusionx?style=for-the-badge">
<img alt="Downloads" src="https://img.shields.io/crates/d/diffusionx?style=for-the-badge">
</p>
</div>

## Implemented

### Random Number Generation

> [!NOTE]
> DiffusionX uses the high-quality [Xoshiro256++](https://prng.di.unimi.it/) random number generator as the common entropy source across all distributions.

- Normal distribution
- Uniform distribution
- Exponential distribution
- Poisson distribution
- $\alpha$-stable distribution

### Stochastic Processes Simulation

- Brownian motion
- $\alpha$-stable Lévy process
- Cauchy process
- $\alpha$-stable subordinator
- Inverse $\alpha$-stable subordinator
- Poisson process
- Fractional Brownian motion
- Continuous-time random walk
- Ornstein-Uhlenbeck process
- Langevin equation
- Generalized Langevin equation
- Subordinated Langevin equation
- Lévy walk
- Birth-death process
- Random walk
- Brownian excursion
- Brownian meander
- Gamma process
- Geometric Brownian motion
- Brownian yet non-Gaussian process

### GPU Acceleration (CUDA/Metal)

The acceleration includes moment calculation and random number generation.

- Brownian motion
- $\alpha$-stable Lévy process
- Ornstein-Uhlenbeck process
- $\alpha$-stable distribution

## Usage

### Random Number Generation

```rust
use diffusionx::random::{normal, uniform, stable};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate a normal random number with mean 0.0 and std 1.0
    let normal_sample = normal::rand(0.0, 1.0)?;
    // Generate 1000 standard normal random numbers
    let std_normal_samples = normal::standard_rands::<f64>(1000);

    // Generate a uniform random number in range [0, 10)
    let uniform_sample = uniform::range_rand(0..10)?;
    // Generate 1000 uniform random numbers in range [0, 1)
    let std_uniform_samples = uniform::standard_rands(1000);

    // Generate 1000 standard stable random numbers
    let stable_samples = stable::standard_rands(1.5, 0.5, 1000)?;

    Ok(())
}
```

### Stochastic Process Simulation

```rust
use diffusionx::simulation::{prelude::*, continuous::Bm};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create standard Brownian motion object
    let bm = Bm::default();
    // Create trajectory with duration 1.0
    let traj = bm.duration(1.0)?;
    // Simulate Brownian motion trajectory with time step 0.01
    let (times, positions) = traj.simulate(0.01)?;
    println!("times: {:?}", times);
    println!("positions: {:?}", positions);

    // Calculate first-order raw moment with 1000 particles and time step 0.01
    let mean = traj.raw_moment(1, 1000, 0.01)?;
    println!("mean: {mean}");
    // Calculate second-order central moment with 1000 particles and time step 0.01
    let msd = traj.central_moment(2, 1000, 0.01)?;
    println!("MSD: {msd}");
    // Calculate EATAMSD with duration 100.0, delta 1.0, 10000 particles, time step 0.1,
    // and Gauss-Legendre quadrature order 10
    let eatamsd = bm.eatamsd(100.0, 1.0, 10000, 0.1, 10)?;
    println!("EATAMSD: {eatamsd}");
    // Calculate first passage time of Brownian motion with boundaries at -1.0 and 1.0
    let fpt = bm.fpt((-1.0, 1.0), 1000, 0.01)?;
    println!("fpt: {fpt}");
    Ok(())
}
```

### Visualization

> [!NOTE]
> The visualization requires the `visualize` feature to be enabled.
> ```toml
> # In your Cargo.toml
> [dependencies]
> diffusionx = { version = "*", features = ["visualize"] }
> ```

```rust
use diffusionx::{
    simulation::{continuous::Bm, prelude::*},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    Ok(())
}
```

### GPU Acceleration

> [!NOTE]
> This requires the `metal` or `cuda` feature to be enabled.
> ```toml
> # In your Cargo.toml
> [dependencies]
> diffusionx = { version = "*", features = ["cuda"] }
> ```

```rust
use diffusionx::{
    simulation::continuous::Bm,
    gpu::GPUMoment,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bm = Bm::<f32>::default();

    // GPU-accelerated moment calculations
    let mean = bm.mean_gpu(1.0, 100_000, 0.01)?;
    let msd = bm.msd_gpu(1.0, 100_000, 0.01)?;
    let raw_moment = bm.raw_moment_gpu(1.0, 2, 100_000, 0.01)?;
    let central_moment = bm.central_moment_gpu(1.0, 2, 100_000, 0.01)?;

    // Fractional moments are also supported
    let frac_raw = bm.frac_raw_moment_gpu(1.0, 1.5, 100_000, 0.01)?;
    let frac_central = bm.frac_central_moment_gpu(1.0, 1.5, 100_000, 0.01)?;

    println!("Mean: {mean}, MSD: {msd}");
    Ok(())
}
```

## Architecture and Extensibility

DiffusionX is designed with a trait-based system for high extensibility and performance:

### Core Traits

- `ContinuousProcess`: Base trait for continuous stochastic processes
- `PointProcess`: Base trait for point processes
- `DiscreteProcess`: Base trait for discrete stochastic processes
- `Moment`: Trait for statistical moments calculation, including (fractional) raw and central moments
- `Visualize`: Trait for plotting process trajectories
- `GPUMoment`: Trait for simulating the (fractional) moments in CUDA.

The `GPUMoment` trait provides GPU-accelerated statistical moment calculations. It is implemented for:
- `Bm<T>` - Brownian Motion
- `OrnsteinUhlenbeck<T>` - Ornstein-Uhlenbeck Process
- `Levy<T>` - Lévy Process

| Method | Description |
|--------|-------------|
| `mean_gpu(duration, particles, time_step)` | Calculate mean (first raw moment) |
| `msd_gpu(duration, particles, time_step)` | Calculate mean squared displacement (second central moment) |
| `raw_moment_gpu(duration, order, particles, time_step)` | Calculate raw moment of integer order |
| `central_moment_gpu(duration, order, particles, time_step)` | Calculate central moment of integer order |
| `frac_raw_moment_gpu(duration, order, particles, time_step)` | Calculate raw moment of fractional order |
| `frac_central_moment_gpu(duration, order, particles, time_step)` | Calculate central moment of fractional order |

### Extending with Custom Processes

1. Adding a New Continuous Process:
   ```rust
   #[derive(Debug, Clone)]
   struct MyProcess {
       // Your parameters
       // Should be `Send + Sync` for parallel computation
       // and `Clone`
   }

   impl ContinuousProcess for MyProcess {
       fn start(&self) -> f64 {
           0.0  // or any default value
       }
       fn simulate(
            &self,
            duration: f64,
            time_step: f64
        ) -> XResult<(Vec<f64>, Vec<f64>)> {
           // Implement your simulation logic
           todo!()
       }
   }
   ```

2. Implementing `ContinuousProcess` trait automatically provides
    - mean `mean`
    - msd `msd`
    - (fractional) raw moment `raw_moment` (`frac_raw_moment`)
    - (fractional) central moment `central_moment` (`frac_central_moment`)
    - first passage time `fpt`
    - occupation time `occupation_time`
    - TAMSD `tamsd`
    - visualization `plot`

The full example implementing the CIR process is [here](./examples/CIR.rs).

## Benchmark

Performance benchmark tests compare the Rust, C++, Julia, and Python implementations, which can be found [here](https://github.com/tangxiangong/diffusionx-benches).

## License

Licensed under either of:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.

