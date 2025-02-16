//! Brownian motion simulation
//!
//! This module provides functions for simulating Brownian motion.
//!

use crate::{SimulationError, XResult, random::normal, simulation::Simulation, utils::cumsum};
use rayon::prelude::*;

use super::{Pair, Params};

/// Brownian motion simulation
///
/// This struct represents a Brownian motion simulation.
///
/// # Fields
///
/// * `start_position` - The starting position of the Brownian motion.
/// * `diffusion_coefficient` - The diffusion coefficient of the Brownian motion.
pub struct Bm {
    start_position: f64,
    diffusion_coefficient: f64,
}

impl Default for Bm {
    fn default() -> Self {
        Self {
            start_position: 0.0,
            diffusion_coefficient: 1.0,
        }
    }
}

impl Bm {
    /// Create a new Brownian motion simulation
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position of the Brownian motion.
    /// * `diffusion_coefficient` - The diffusion coefficient of the Brownian motion.
    pub fn new(
        start_position: impl Into<f64>,
        diffusion_coefficient: impl Into<f64>,
    ) -> XResult<Self> {
        let start_position = start_position.into();
        let diffusion_coefficient = diffusion_coefficient.into();
        if diffusion_coefficient <= 0.0 {
            return Err(SimulationError::InvalidParameters(
                "diffusion_coefficient must be positive".to_string(),
            )
            .into());
        }
        Ok(Self {
            start_position,
            diffusion_coefficient,
        })
    }
}

/// impl `Simulation` trait for Brownian motion
impl Simulation for Bm {
    type Time = f64;
    type Position = f64;
    /// Simulate Brownian motion
    ///
    /// This method simulates Brownian motion.
    ///
    /// # Returns
    ///
    /// The result of the Brownian motion simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let bm = Bm::new(10.0, 1.0).unwrap();
    /// let params = ParamsBuilder::default().time_step(0.1).duration(1).build().unwrap();
    /// let (t, x) = bm.simulate(params).unwrap();
    /// ```
    fn simulate(&self, params: Params) -> XResult<Pair<Self::Time, Self::Position>> {
        let tau = params.time_step()?;
        let duration = params.duration()?;
        simulate_bm(
            self.start_position,
            self.diffusion_coefficient,
            tau,
            duration,
        )
    }
}

/// Simulate Brownian motion
///
/// This function simulates Brownian motion.
///
/// # Arguments
///
/// * `start_position` - The starting position of the Brownian motion.  
/// * `diffusion_coefficient` - The diffusion coefficient of the Brownian motion.
/// * `tau` - The time step of the Brownian motion.
/// * `duration` - The duration of the Brownian motion.
///
/// # Returns
///
/// The result of the Brownian motion simulation.   
/// 
/// # Example
///
/// ```rust
/// let bm = Bm::new(10.0, 1.0).unwrap();
/// let params = ParamsBuilder::default().time_step(0.1).duration(1).build().unwrap();
/// let (t, x) = bm.simulate(params).unwrap();
/// ```
pub fn simulate_bm(
    start_position: impl Into<f64>,
    diffusion_coefficient: impl Into<f64>,
    tau: impl Into<f64>,
    duration: impl Into<f64>,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let start_position = start_position.into();
    let diffusion_coefficient = diffusion_coefficient.into();
    let tau = tau.into();
    let duration = duration.into();
    let num_steps = (duration / tau).ceil() as usize;
    let t = (0..=num_steps)
        .into_par_iter()
        .map(|i| tau * i as f64)
        .collect::<Vec<_>>();
    let noise = normal::rands(0.0, 2.0 * diffusion_coefficient * tau, num_steps)?;
    let x = cumsum(start_position, &noise);
    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use crate::simulation::ParamsBuilder;

    use super::*;

    #[test]
    fn test_simulate_bm() {
        let bm = Bm::new(10.0, 1.0).unwrap();
        let params = ParamsBuilder::default()
            .time_step(0.1)
            .duration(1)
            .build()
            .unwrap();
        let (t, x) = bm.simulate(params).unwrap();
        println!("t: {:?}", t);
        println!("x: {:?}", x);
    }
}
