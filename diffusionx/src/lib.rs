//! DiffusionX is a multi-threaded high-performance Rust library for random number/stochastic process simulation.
//! # Usage
//! ```rust
//! use diffusionx::simulation::{Bm, Simulation, Functional};
//! // Brownian motion simulation
//! let bm = Bm::new(0.0, 1.0, 1.0)?;  // Create Brownian motion object: initial position 0, diffusion coefficient 1, duration 1
//! let time_step = 0.01;  // Time step
//! let (times, positions) = bm.simulate(time_step)?;  // Simulate Brownian motion trajectory
//! // Monte Carlo simulation of Brownian motion statistics
//! let mean = bm.mean(time_step, 1000)?;  // Mean  bm.raw_moment(time_step, 1, 1000)?;
//! let msd = bm.msd(time_step, 1000)?;  // Mean square displacement  bm.central_moment(time_step, 2, 1000)?;
//! // First passage time of Brownian motion
//! let max_duration = 1000; // if over this duration, the simulation will be terminated and return None
//! let fpt = bm.fpt(time_step, (-1.0, 1.0), max_duration)?;
//! ```
//! # Progress
//! ## Random Number Generation
//! - [x] Normal distribution
//! - [x] Uniform distribution
//! - [x] Exponential distribution
//! - [x] Poisson distribution
//! - [x] Alpha-stable distribution
//! ## Stochastic Processes
//! - [x] Brownian motion
//! - [ ] Alpha-stable Lévy process
//! - [ ] Fractional Brownian motion
//! - [ ] Poisson process
//! - [ ] Compound Poisson process
//! - [ ] Langevin equation
//! # License
//! This project is dual-licensed under:
//! * [MIT License](https://opensource.org/licenses/MIT)
//! * [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
//!
//! You can choose to use either license.

mod error;
pub use error::*;
pub mod random;
pub mod simulation;
pub mod utils;
