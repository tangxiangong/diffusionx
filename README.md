# DiffusionX

English | [简体中文](README-zh.md)
> DiffusionX is a multi-threaded high-performance Rust library for random number generation and stochastic process simulation.

[![docs.rs](https://img.shields.io/badge/docs.rs-latest-blue.svg)](https://docs.rs/diffusionx/latest/diffusionx/)
[![crates.io](https://img.shields.io/crates/v/diffusionx.svg)](https://crates.io/crates/diffusionx)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

## Implemented

### Random Number Generation

- [x] Normal distribution
- [x] Uniform distribution
- [x] Exponential distribution
- [x] Poisson distribution
- [x] $\alpha$-stable distribution

### Stochastic Processes Simulation

- [x] Brownian motion
- [x] $\alpha$-stable Lévy process
- [x] Cauchy process
- [x] $\alpha$-stable subordinator
- [x] Inverse $\alpha$-stable subordinator
- [x] Poisson process
- [x] Fractional Brownian motion
- [x] Continuous-time random walk
- [x] Ornstein-Uhlenbeck process
- [x] Langevin equation
- [x] Generalized Langevin equation
- [x] Subordinated Langevin equation
- [x] Lévy walk
- [x] Birth-death process
- [x] Random walk
- [x] Brownian excursion
- [x] Brownian meander
- [x] Gamma process


## Usage

### Random Number Generation

```rust
use diffusionx::random::{normal, uniform, stable};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate a normal random number with mean 0.0 and std 1.0
    let normal_sample = normal::rand(0.0, 1.0)?;
    // Generate 1000 standard normal random numbers
    let std_normal_samples = normal::standard_rands(1000);

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
    println!("mean: {:?}", mean);
    // Calculate second-order central moment with 1000 particles and time step 0.01
    let msd = traj.central_moment(2, 1000, 0.01)?;
    println!("msd: {:?}", msd);
    // Calculate TAMSD with duration 100.0, delta 1.0, 10000 particles, time step 0.1, and Gauss-Legendre quadrature order 10
    let tamsd = bm.tamsd(100.0, 1.0, 10000, 0.1, 10)?;
    println!("tamsd: {:?}", tamsd);
    // Calculate first passage time of Brownian motion with boundaries at -1.0 and 1.0
    let fpt = bm.fpt((-1.0, 1.0), 1000, 0.01)?;
    println!("fpt: {:?}", fpt);
    Ok(())
}
```

### Visualization Example

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
   // Calculate first passage time with time step 0.01,
   // boundaries at -1.0 and 1.0, and 1000 particles
   let fpt = bm.fpt(0.01, (-1.0, 1.0), 1000)?;
   ```

2. **Occupation Time**: Measure time spent by a process in a specified region
   ```rust
   // For a Brownian motion process
   let bm = Bm::default();
   let traj = bm.duration(10.0)?;
   // Calculate time spent in region [0.0, 2.0] with time step 0.01
   let occupation = traj.occupation_time(0.01, (0.0, 2.0))?;
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

2. Implementing `ContinuousProcess` trait automatically provides
    - mean `mean`
    - msd `msd`
    - raw moment `raw_moment`
    - central moment `central_moment`
    - first passage time `fpt`
    - occupation time `occupation_time`
    - TAMSD `tamsd`
    - visualization `plot`

**Example:**

```rust
use diffusionx::{XError, XResult, random::normal, simulation::prelude::*, utils::write_csv};

/// CIR
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
struct CIR {
    speed: f64,
    mean: f64,
    volatility: f64,
    start_position: f64,
}

impl CIR {
    fn new(
        speed: impl Into<f64>,
        mean: impl Into<f64>,
        volatility: impl Into<f64>,
        start_position: impl Into<f64>,
    ) -> XResult<Self> {
        let speed: f64 = speed.into();
        if speed <= 0.0 {
            return Err(XError::InvalidParameters(format!(
                "speed must be greater than 0, but got {}",
                speed
            )));
        }
        Ok(Self {
            speed,
            mean: mean.into(),
            volatility: volatility.into(),
            start_position: start_position.into(),
        })
    }
}

impl ContinuousProcess for CIR {
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        let duration = duration.into();
        let num_steps = (duration / time_step).ceil() as usize;

        let initial_x = self.start_position.max(0.0);
        let noises = normal::standard_rands(num_steps);

        let t: Vec<f64> = (0..=num_steps).map(|i| i as f64 * time_step).collect();

        let x = std::iter::once(initial_x)
            .chain((0..num_steps).scan(initial_x, |state, i| {
                let current_x = *state;
                let drift = self.speed * (self.mean - current_x);
                let diffusion = self.volatility * current_x.sqrt().max(0.0);

                let next_x =
                    current_x + drift * time_step + diffusion * noises[i] * time_step.sqrt();
                *state = next_x.max(0.0);

                Some(*state)
            }))
            .collect();

        Ok((t, x))
    }
}

fn main() -> XResult<()> {
    let duration = 10;
    let particles = 10_000;
    let time_step = 0.01;
    let cir = CIR::new(1, 1, 1, 0.5)?;
    let traj = cir.duration(duration)?;
    let (t, x) = cir.simulate(duration, time_step)?;
    write_csv("tmp/CIR.csv", &t, &x)?;
    // mean
    let mean = cir.mean(duration, particles, time_step)?; // or let mean = traj.raw_moment(1, particles, time_step)?;
    println!("mean: {:?}", mean);
    // msd
    let msd = cir.msd(duration, particles, time_step)?; // or let msd = traj.central_moment(2, particles, time_step)?;
    println!("MSD: {:?}", msd);
    // FPT
    let max_duration = 1000;
    let fpt = cir.fpt((-1, 1), max_duration, time_step)?.unwrap_or(-1.0);
    println!("FPT: {:?}", fpt);
    // occupation time
    let occupation_time = cir.occupation_time((-1, 1), duration, time_step)?;
    println!("Occupation Time: {:?}", occupation_time);
    // TAMSD
    let slag = 1;
    let quad_order = 10;
    let tamsd = cir.tamsd(duration, slag, particles, time_step, quad_order)?;
    println!("TAMSD: {:?}", tamsd);

    let config = PlotConfigBuilder::default()
        .time_step(time_step)
        .output_path("tmp/CIR.svg")
        .caption("CIR")
        .x_label("t")
        .y_label("r")
        .legend("CIR")
        .backend(PlotterBackend::SVG)
        .build()
        .unwrap();
    traj.plot(&config)?;
    Ok(())
}
```

## Benchmark
The related content can be found in the **Benchmark** section of [py-diffusionx](https://github.com/tangxiangong/py-diffusionx).

## License

This project is dual-licensed under:

* [MIT License](https://opensource.org/licenses/MIT)
* [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

You can choose to use either license.
