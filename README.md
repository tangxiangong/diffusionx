<div align=center>
<h1 aligh="center">
DiffusionX
</h1>
<p align="center">
A multi-threaded high-performance Rust library for random number generation and stochastic process simulation.
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
- [x] Geometric Brownian motion
- [x] Brownian yet non-Gaussian process


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
> diffusionx = { version = "0.5", features = ["visualize"] }
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

## Architecture and Extensibility

DiffusionX is designed with a trait-based system for high extensibility and performance:

### Core Traits

- `ContinuousProcess`: Base trait for continuous stochastic processes
- `PointProcess`: Base trait for point processes
- `DiscreteProcess`: Base trait for discrete stochastic processes
- `Moment`: Trait for statistical moments calculation, including raw and central moments
- `Visualize`: Trait for plotting process trajectories

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
    - raw moment `raw_moment`
    - central moment `central_moment`
    - first passage time `fpt`
    - occupation time `occupation_time`
    - TAMSD `tamsd`
    - visualization `plot`

**Example:**

```rust
use diffusionx::{
    XError, XResult,
    random::normal,
    simulation::prelude::*,
    utils::{diff, linspace, write_csv},
};

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
                "speed must be greater than 0, but got {speed}",
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
    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        let t = linspace(0.0, duration, time_step);
        let num_steps = t.len() - 1;
        let initial_x = self.start_position.max(0.0);
        let noises = normal::standard_rands::<f64>(num_steps);
        let delta = diff(&t);

        let x = std::iter::once(initial_x)
            .chain(
                noises
                    .iter()
                    .zip(delta)
                    .scan(initial_x, |state, (&xi, delta_t)| {
                        let current_x = *state;
                        let drift = self.speed * (self.mean - current_x);
                        let diffusion = self.volatility * current_x.sqrt().max(0.0);

                        let next_x = current_x + drift * delta_t + diffusion * xi * delta_t.sqrt();
                        *state = next_x.max(0.0);

                        Some(*state)
                    }),
            )
            .collect();

        Ok((t, x))
    }
}

fn main() -> XResult<()> {
    let duration = 10.0;
    let particles = 10_000;
    let time_step = 0.01;
    let cir = CIR::new(1, 1, 1, 0.5)?;
    let traj = cir.duration(duration)?;
    let (t, x) = cir.simulate(duration, time_step)?;
    write_csv("tmp/CIR.csv", &t, &x)?;
    // mean
    let mean = cir.mean(duration, particles, time_step)?; // or let mean = traj.raw_moment(1, particles, time_step)?;
    println!("mean: {mean}");
    // msd
    let msd = cir.msd(duration, particles, time_step)?; // or let msd = traj.central_moment(2, particles, time_step)?;
    println!("MSD: {msd}");
    // FPT
    let max_duration = 1000.0;
    let fpt = cir
        .fpt((-1.0, 1.0), max_duration, time_step)?
        .unwrap_or(-1.0);
    println!("FPT: {fpt}");
    // occupation time
    let occupation_time = cir.occupation_time((-1.0, 1.0), duration, time_step)?;
    println!("Occupation Time: {occupation_time}");
    // TAMSD
    let slag = 1.0;
    let quad_order = 10;
    let tamsd = TAMSD::new(&cir, duration, slag)?;
    let eatamsd = tamsd.mean(particles, time_step, quad_order)?;
    println!("EATAMSD: {eatamsd}");

    // Visualization
    let config = PlotConfigBuilder::default()
        .time_step(time_step)
        .output_path("tmp/CIR.svg")
        .caption("CIR")
        .show_grid(false)
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
**Result:**
```
mean: 0.9957644815350275
MSD: 0.7441251895881059
FPT: 0.38
Occupation Time: 4.719999999999995
EATAMSD: 0.6085042089895467
```
<img src="https://raw.githubusercontent.com/tangxiangong/diffusionx/dev/assets/CIR.svg" alt="CIR"/>

## License

Licensed under either of:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
