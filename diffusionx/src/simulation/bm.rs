//! Brownian motion simulation
//!
//! This module provides functions for simulating Brownian motion.
//!

use crate::{random::normal, simulation::Simulation, utils::cumsum, SimulationError, XResult};
use rayon::prelude::*;

use super::{MomentMC, Pair, Params};

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

    fn get_params(&self, params: &Params) -> XResult<(f64, f64, f64, f64, usize)> {
        let start_position = self.start_position;
        let diffusion_coefficient = self.diffusion_coefficient;
        let tau = params.time_step()?;
        let duration = params.duration()?;
        let num_steps = (duration / tau).ceil() as usize;
        Ok((start_position, diffusion_coefficient, tau, duration, num_steps))
    }

    pub fn mean(&self, params: Params, particles: usize) -> XResult<f64> {
        self.raw_moment(params, 1, particles)
    }

    pub fn msd(&self, params: Params, particles: usize) -> XResult<f64> {
        self.central_moment(params, 2, particles)
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
        let (start_position, diffusion_coefficient, tau, duration, _) = self.get_params(&params)?;
        simulate_bm(
            start_position,
            diffusion_coefficient,
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

impl MomentMC for Bm {
    fn raw_moment(&self, params: Params, order: i32, particles: usize) -> XResult<f64> {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(
                "particles must be positive".to_string(),
            )
            .into());
        }
        if order < 0 {
            return Err(SimulationError::InvalidParameters(
                "order must be non-negative".to_string(),
            )
            .into());
        }
        if order == 0 {
            return Ok(0.0);
        }

        let (start_position, diffusion_coefficient, tau, duration, num_steps) = self.get_params(&params)?;
        let result = (0..particles).into_par_iter()
        .map(|_| -> XResult<f64> {
            let (_, x) = simulate_bm(start_position, diffusion_coefficient, tau, duration)?;
            let end_position = x.get(num_steps);
            match end_position {
                Some(position) => Ok(position.powi(order)),
                None => Err(SimulationError::Unknown.into()),
            }
        })
        .try_fold(
            || 0.0,
            |acc, res| {
                res.map(|v| acc + v)
            }
        )
        .try_reduce(|| 0.0, |a, b| Ok(a + b))? / particles as f64;
        Ok(result)
    }
    fn central_moment(&self, params: Params, order: i32, particles: usize) -> XResult<f64> {
        let mean = self.raw_moment(params, 1, particles)?;
        let (start_position, diffusion_coefficient, tau, duration, num_steps) = self.get_params(&params)?;
        let result = (0..particles).into_par_iter()
        .map(|_| -> XResult<f64> {
            let (_, x) = simulate_bm(start_position, diffusion_coefficient, tau, duration)?;
            let end_position = x.get(num_steps);
            match end_position {
                Some(position) => Ok((position-mean).powi(order)),
                None => Err(SimulationError::Unknown.into()),
            }
        })
        .try_fold(
            || 0.0,
            |acc, res| {
                res.map(|v| acc + v)
            }
        )
        .try_reduce(|| 0.0, |a, b| Ok(a + b))? / particles as f64;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::simulation::{ParamsBuilder, MomentMC};

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

    #[test]
    fn test_raw_moment() {
        let bm = Bm::new(10.0, 1.0).unwrap();
        let params = ParamsBuilder::default()
            .time_step(0.1)
            .duration(10)
            .build()
            .unwrap();
        let moment = bm.raw_moment(params, 1, 1000).unwrap();
        println!("moment: {:?}", moment);
    }
}
