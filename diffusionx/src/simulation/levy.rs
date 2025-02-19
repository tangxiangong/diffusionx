//! Levy process simulation
//! For Brownian motion, see [`crate::simulation::bm`].

use crate::{
    SimulationError, XResult,
    random::stable,
    simulation::{Pair, Simulation, Stochastic, functional::FirstPassageTime},
    utils::cumsum,
};
use rayon::prelude::*;

/// Levy process simulation
///
/// This struct represents a Levy process simulation.
///
/// # Fields
///
/// * `start_position` - The starting position of the Levy process.
/// * `alpha` - The stability index of the Levy process.
#[derive(Debug, Clone)]
pub struct Levy {
    start_position: f64,
    alpha: f64,
}

impl Levy {
    /// Create a new Levy process simulation
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position of the Levy process.
    /// * `alpha` - The stability index of the Levy process.
    pub fn new(start_position: impl Into<f64>, alpha: impl Into<f64>) -> XResult<Self> {
        let start_position = start_position.into();
        let alpha = alpha.into();
        if alpha <= 0.0 || alpha > 2.0 {
            return Err(SimulationError::InvalidParameters(
                "alpha must be in the range (0, 2]".to_string(),
            )
            .into());
        }
        Ok(Self {
            start_position,
            alpha,
        })
    }

    /// Get the starting position of the Levy process simulation
    pub fn start_position(&self) -> f64 {
        self.start_position
    }

    /// Get the stability index of the Levy process simulation
    pub fn index(&self) -> f64 {
        self.alpha
    }

    /// Get the first passage time of the Levy process simulation
    /// 
    /// # Arguments
    /// 
    /// * `domain` - The domain of the Levy process simulation.
    /// * `max_duration` - The maximum duration of the Levy process simulation.
    /// * `time_step` - The time step of the Levy process simulation.
    ///
    /// # Returns
    ///
    /// `Option<f64>`
    /// * None if the first passage time is not found within the maximum duration.
    /// * A f64 representing the first passage time of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let levy = Levy::new(0.0, 1.5).unwrap();
    /// let params = 0.1;
    /// let (t, x) = levy.simulate(params).unwrap();
    /// ```
    pub fn fpt(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        max_duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<Option<f64>> {
        let fpt = FirstPassageTime::new(self, domain)?;
        fpt.simulate(max_duration, time_step)
    }
}

impl Stochastic for Levy {}

/// impl `Simulation` trait for Levy process
impl Simulation for Levy {
    /// Simulate Levy process
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the Levy process simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the Levy process simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let levy = Levy::new(0.0, 1.5).unwrap();
    /// let params = 0.1;
    /// let (t, x) = levy.simulate(params).unwrap();
    /// ```
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        if time_step <= 0.0 {
            return Err(SimulationError::InvalidParameters(
                "time_step must be positive".to_string(),
            )
            .into());
        }
        let duration = duration.into();
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(
                "duration must be positive".to_string(),
            )
            .into());
        }
        simulate_levy(self.start_position, self.alpha, duration, time_step)
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
/// let levy = Levy::new(0.0, 1.5).unwrap();
/// let params = 0.1;
/// let (t, x) = levy.simulate(params).unwrap();
/// ```
pub fn simulate_levy(
    start_position: impl Into<f64>,
    alpha: impl Into<f64>,
    duration: impl Into<f64>,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let start_position = start_position.into();
    let alpha = alpha.into();
    let duration = duration.into();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_levy() {
        let levy = Levy::new(10.0, 1.5).unwrap();
        let time_step = 0.1;
        let duration = 1.0;
        let (t, x) = levy.simulate(duration, time_step).unwrap();
        println!("t: {:?}", t);
        println!("x: {:?}", x);
    }

    #[test]
    fn test_fpt() {
        let levy = Levy::new(0.0, 1.5).unwrap();
        let time_step = 0.1;
        let fpt = levy.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {:?}", fpt);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Levy>();
    }
}
