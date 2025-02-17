//! Levy process simulation
//!
//! This module provides functions for simulating Levy process.
//!

use crate::{SimulationError, XResult, random::stable, simulation::Simulation, utils::cumsum};
use rayon::prelude::*;

use super::{CheckedParams, Functional, Pair};

/// Levy process simulation
///
/// This struct represents a Levy process simulation.
///
/// # Fields
///
/// * `start_position` - The starting position of the Levy process.
/// * `alpha` - The stability index of the Levy process.
/// * `duration` - The duration of the Levy process.
#[derive(Debug, Clone)]
pub struct Levy {
    start_position: f64,
    alpha: f64,
    duration: f64,
}

impl Levy {
    /// Create a new Levy process simulation
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position of the Levy process.
    /// * `alpha` - The stability index of the Levy process.
    /// * `duration` - The duration of the Levy process.
    pub fn new(
        start_position: impl Into<f64>,
        alpha: impl Into<f64>,
        duration: impl Into<f64>,
    ) -> XResult<Self> {
        let start_position = start_position.into();
        let alpha = alpha.into();
        let duration = duration.into();
        if alpha <= 0.0 || alpha > 2.0 {
            return Err(SimulationError::InvalidParameters(
                "alpha must be in the range (0, 2]".to_string(),
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
            alpha,
            duration,
        })
    }

    pub fn build(&self) -> XResult<Self> {
        self.check_params(0.01)?;
        Ok(self.clone())
    }

    pub fn get_params(&self) -> (f64, f64, f64) {
        (self.start_position, self.alpha, self.duration)
    }
}

/// impl `Simulation` trait for Levy process
impl Simulation for Levy {
    fn get_duration(&self) -> f64 {
        self.duration
    }

    fn mut_duration(&mut self, duration: f64) {
        self.duration = duration;
    }
    /// Simulate Levy process
    ///
    /// This method simulates Levy process.
    ///
    /// # Returns
    ///
    /// The result of the Levy process simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let levy = Levy::new(0.0, 1.5, 1.0).unwrap();
    /// let params = 0.1;
    /// let (t, x) = levy.simulate(params).unwrap();
    /// ```
    fn simulate_check(&self, time_step: f64) -> XResult<Pair> {
        self.check_params(time_step)?;
        self.simulate(time_step)
    }

    fn simulate(&self, time_step: f64) -> XResult<Pair> {
        simulate_levy(self.start_position, self.alpha, time_step, self.duration)
    }
}

/// Simulate Levy process
///
/// This function simulates Levy process.
///
/// # Arguments
///
/// * `start_position` - The starting position of the Levy process.  
/// * `alpha` - The stability index of the Levy process.
/// * `tau` - The time step of the Levy process.
/// * `duration` - The duration of the Levy process.
///
/// # Returns
///
/// The result of the Levy process simulation.   
///
/// # Example
///
/// ```rust
/// let levy = Levy::new(0.0, 1.5, 1.0).unwrap();
/// let params = 0.1;
/// let (t, x) = levy.simulate(params).unwrap();
/// ```
pub fn simulate_levy(
    start_position: impl Into<f64>,
    alpha: impl Into<f64>,
    time_step: f64,
    duration: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let start_position = start_position.into();
    let alpha = alpha.into();
    let num_steps = (duration / time_step).ceil() as usize;
    let t = (0..=num_steps)
        .into_par_iter()
        .map(|i| time_step * i as f64)
        .collect::<Vec<_>>();
    let noise = stable::standard_rands(alpha, 0.0, num_steps)?
        .into_par_iter()
        .map(|x| x * time_step.powf(1.0 / alpha))
        .collect::<Vec<_>>();
    let x = cumsum(start_position, &noise);
    Ok((t, x))
}

// impl Moment for Bm {}

impl Functional for Levy {}

impl CheckedParams for Levy {
    fn check_params(&self, time_step: f64) -> XResult<()> {
        if self.alpha <= 0.0 || self.alpha > 2.0 {
            return Err(SimulationError::InvalidParameters(
                "alpha must be in the range (0, 2]".to_string(),
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
    use crate::simulation::Functional;

    use super::*;

    #[test]
    fn test_simulate_levy() {
        let levy = Levy::new(10.0, 1.5, 1.0).unwrap();
        let time_step = 0.1;
        let (t, x) = levy.simulate(time_step).unwrap();
        println!("t: {:?}", t);
        println!("x: {:?}", x);
    }

    // #[test]
    // fn test_raw_moment() {
    //     let levy = Levy::new(10.0, 1.5, 1.0).unwrap();
    //     let time_step = 0.1;
    //     let moment = levy.raw_moment(time_step, 1, 1000).unwrap();
    //     println!("moment: {:?}", moment);
    // }

    #[test]
    fn test_fpt() {
        let levy = Levy::new(0.0, 1.5, 1.0).unwrap();
        let time_step = 0.1;
        let fpt = levy.fpt(time_step, (-1.0, 1.0), 1000).unwrap();
        println!("fpt: {:?}", fpt);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Levy>();
    }
}
