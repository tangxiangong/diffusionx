//! Gamma process simulation
//!

use crate::{SimulationError, XResult, random::gamma, simulation::prelude::*, utils::cumsum};
use rayon::prelude::*;

/// Gamma process
///
/// This struct represents an Gamma process.
///
/// # Mathematical Formulation
///
/// A Gamma process is a process that is non-negative and has a non-decreasing sample path with a Gamma distribution.
///
/// # Fields
///
/// * `shape` - The shape parameter of the Gamma distribution.
/// * `rate` - The rate parameter of the Gamma distribution.
#[derive(Debug, Clone)]
pub struct Gamma {
    shape: f64,
    rate: f64,
}

impl Gamma {
    /// Create a new Gamma process
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape parameter of the Gamma distribution.
    /// * `rate` - The rate parameter of the Gamma distribution.
    pub fn new(shape: impl Into<f64>, rate: impl Into<f64>) -> XResult<Self> {
        let shape = shape.into();
        let rate = rate.into();
        if shape <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `shape` must be positive, got {}",
                shape
            ))
            .into());
        }
        if rate <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `rate` must be positive, got {}",
                rate
            ))
            .into());
        }
        Ok(Self { shape, rate })
    }

    /// Get the shape parameter of the Gamma distribution
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Get the rate parameter of the Gamma distribution
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Get the first passage time of the Gamma process
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the Gamma process simulation.
    /// * `max_duration` - The maximum duration of the Gamma process simulation.
    /// * `time_step` - The time step of the Gamma process simulation.
    ///
    /// # Returns
    ///
    /// `Option<f64>`
    /// * None if the first passage time is not found within the maximum duration.
    /// * A f64 representing the first passage time of the simulation.
    pub fn fpt(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        max_duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<Option<f64>> {
        let fpt = FirstPassageTime::new(self, domain)?;
        fpt.simulate(max_duration, time_step)
    }

    /// Get the occupation time of the subordinator
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the subordinator.
    /// * `duration` - The duration of the subordinator.
    /// * `time_step` - The time step of the subordinator.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::Gamma;
    /// let gamma = Gamma::new(0.5, 1.0).unwrap();
    /// let ot = gamma.occupation_time((-1.0, 1.0), 10.0, 0.1).unwrap();
    /// ```
    pub fn occupation_time(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<f64> {
        let ot = OccupationTime::new(self, domain, duration)?;
        ot.simulate(time_step)
    }
}

/// impl `ContinuousProcess` trait for Gamma
impl ContinuousProcess for Gamma {
    /// Simulate Gamma process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the Gamma process simulation.
    /// * `time_step` - The time step of the Gamma process simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::Gamma;
    /// let gamma = Gamma::new(0.5, 1.0).unwrap();
    /// let (t, x) = gamma.simulate(1.0, 0.1).unwrap();
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
        simulate_gamma(self.shape, self.rate, duration, time_step)
    }
}

/// Simulate Gamma process
///
/// This function simulates Gamma process.
///
/// # Arguments
///
/// * `shape` - The shape parameter of the Gamma distribution.
/// * `rate` - The rate parameter of the Gamma distribution.
/// * `duration` - The duration of the Gamma process simulation.
/// * `time_step` - The time step of the Gamma process simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::gamma::simulate_gamma;
/// let (t, x) = simulate_gamma(0.5, 1.0, 1.0, 0.1).unwrap();
/// ```
pub fn simulate_gamma(
    shape: f64,
    rate: f64,
    duration: impl Into<f64>,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let duration = duration.into();
    let num_steps = (duration / time_step).ceil() as usize;
    let t = (0..=num_steps)
        .into_par_iter()
        .map(|i| time_step * i as f64)
        .collect::<Vec<_>>();
    let scale = 1.0 / rate;
    let noise = gamma::rands(shape * time_step, scale, num_steps)?;
    let x = cumsum(0.0, &noise);
    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_gamma() {
        let gamma = Gamma::new(0.5, 1.0).unwrap();
        let time_step = 0.1;
        let duration = 1.0;
        let (t, x) = gamma.simulate(duration, time_step).unwrap();
        println!("t: {:?}", t);
        println!("x: {:?}", x);
    }

    #[test]
    fn test_fpt() {
        let gamma = Gamma::new(0.5, 1.0).unwrap();
        let time_step = 0.1;
        let fpt = gamma.fpt((-1.0, 1.0), 1000.0, time_step).unwrap().unwrap();
        println!("fpt: {:?}", fpt);
    }

    #[test]
    fn test_occupation_time() {
        let gamma = Gamma::new(0.5, 1.0).unwrap();
        let time_step = 0.1;
        let ot = gamma.occupation_time((-1.0, 1.0), 10.0, time_step).unwrap();
        println!("ot: {:?}", ot);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Gamma>();
    }
}
