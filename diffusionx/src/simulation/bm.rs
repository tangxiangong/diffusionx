//! Brownian motion simulation
//!
//! This module provides functions for simulating Brownian motion.
//!

use crate::{SimulationError, XResult, random::normal, simulation::Simulation, utils::cumsum};
use rayon::prelude::*;

use super::{CheckedParams, Moment, Pair};

/// Brownian motion simulation
///
/// This struct represents a Brownian motion simulation.
///
/// # Fields
///
/// * `start_position` - The starting position of the Brownian motion.
/// * `diffusion_coefficient` - The diffusion coefficient of the Brownian motion.
/// * `duration` - The duration of the Brownian motion.
#[derive(Debug, Clone)]
pub struct Bm {
    start_position: f64,
    diffusion_coefficient: f64,
    duration: f64,
}

impl Default for Bm {
    fn default() -> Self {
        Self {
            start_position: 0.0,
            diffusion_coefficient: 1.0,
            duration: 1.0,
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
        duration: impl Into<f64>,
    ) -> XResult<Self> {
        let start_position = start_position.into();
        let diffusion_coefficient = diffusion_coefficient.into();
        let duration = duration.into();
        if diffusion_coefficient <= 0.0 {
            return Err(SimulationError::InvalidParameters(
                "diffusion_coefficient must be positive".to_string(),
            )
            .into());
        }
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(
                "duration must be positive".to_string(),
            )
            .into());
        }
        Ok(Self {
            start_position,
            diffusion_coefficient,
            duration,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn new_unchecked(
        start_position: f64,
        diffusion_coefficient: f64,
        duration: f64,
    ) -> Self {
        Self {
            start_position,
            diffusion_coefficient,
            duration,
        }
    }

    pub fn get_params(&self) -> (f64, f64, f64) {
        (
            self.start_position,
            self.diffusion_coefficient,
            self.duration,
        )
    }

    pub fn mean(&self, time_step: f64, particles: usize) -> XResult<f64> {
        self.raw_moment(time_step, 1, particles)
    }

    pub fn msd(&self, time_step: f64, particles: usize) -> XResult<f64> {
        self.central_moment(time_step, 2, particles)
    }
}

/// impl `Simulation` trait for Brownian motion
impl Simulation for Bm {
    fn get_duration(&self) -> f64 {
        self.duration
    }

    fn mut_duration(&mut self, duration: f64) {
        self.duration = duration;
    }
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
    /// let params = 0.1;
    /// let (t, x) = bm.simulate(params).unwrap();
    /// ```
    fn simulate_check(&self, time_step: f64) -> XResult<Pair> {
        self.check_params(time_step)?;
        simulate_bm(
            self.start_position,
            self.diffusion_coefficient,
            time_step,
            self.duration,
        )
    }

    fn simulate(&self, time_step: f64) -> XResult<Pair> {
        simulate_bm(
            self.start_position,
            self.diffusion_coefficient,
            time_step,
            self.duration,
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
    time_step: f64,
    duration: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let start_position = start_position.into();
    let diffusion_coefficient = diffusion_coefficient.into();
    let num_steps = (duration / time_step).ceil() as usize;
    let t = (0..=num_steps)
        .into_par_iter()
        .map(|i| time_step * i as f64)
        .collect::<Vec<_>>();
    let noise = normal::rands(0.0, 2.0 * diffusion_coefficient * time_step, num_steps)?;
    let x = cumsum(start_position, &noise);
    Ok((t, x))
}

impl CheckedParams for Bm {
    fn check_params(&self, time_step: f64) -> XResult<()> {
        if self.diffusion_coefficient <= 0.0 {
            return Err(SimulationError::InvalidParameters(
                "diffusion_coefficient must be positive".to_string(),
            )
            .into());
        }
        if self.duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(
                "duration must be positive".to_string(),
            )
            .into());
        }
        if time_step <= 0.0 {
            return Err(SimulationError::InvalidParameters(
                "time_step must be positive".to_string(),
            )
            .into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::simulation::{Functional, Moment};

    use super::*;

    #[test]
    fn test_simulate_bm() {
        let bm = Bm::new(10.0, 1.0, 1.0).unwrap();
        let time_step = 0.1;
        let (t, x) = bm.simulate(time_step).unwrap();
        println!("t: {:?}", t);
        println!("x: {:?}", x);
    }

    #[test]
    fn test_raw_moment() {
        let bm = Bm::new(10.0, 1.0, 1.0).unwrap();
        let time_step = 0.1;
        let moment = bm.raw_moment(time_step, 1, 1000).unwrap();
        println!("moment: {:?}", moment);
    }

    #[test]
    fn test_fpt() {
        let bm = Bm::new(0.0, 1.0, 1.0).unwrap();
        let time_step = 0.1;
        let fpt = bm.fpt(time_step, (-1.0, 1.0), 1000).unwrap();
        println!("fpt: {:?}", fpt);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Bm>();
    }
}
