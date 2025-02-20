//! Subordinator simulation
//! 
//! An alpha stable subordinator is an alpha stable Levy process that is non-negative and has a non-decreasing sample path.
//! For Levy process, see [`crate::simulation::levy`].

use crate::{
    SimulationError, XResult,
    random::stable,
    simulation::prelude::*,
    utils::cumsum,
};
use rayon::prelude::*;

/// Subordinator simulation
///
/// This struct represents a subordinator simulation.
///
/// # Fields
///
/// * `alpha` - The stability index of the subordinator, whose value must be in the range (0, 1).
#[derive(Debug, Clone)]
pub struct Subordinator {
    alpha: f64,
}

impl Subordinator {
    /// Create a new subordinator simulation
    ///
    /// # Arguments
    ///
    /// * `alpha` - The stability index of the subordinator, whose value must be in the range (0, 1).
    pub fn new(alpha: f64) -> XResult<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(SimulationError::InvalidParameters(
                "alpha must be in the range (0, 1)".to_string(),
            )
            .into());
        }
        Ok(Self {
            alpha,
        })
    }

    /// Get the stability index of the subordinator
    pub fn index(&self) -> f64 {
        self.alpha
    }

    /// Get the first passage time of the subordinator
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the subordinator simulation.
    /// * `max_duration` - The maximum duration of the subordinator simulation.
    /// * `time_step` - The time step of the subordinator simulation.
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
    /// let subordinator = Subordinator::new(0.5).unwrap();
    /// let params = 0.1;
    /// let (t, x) = subordinator.simulate(params).unwrap();
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

    /// Get the occupation time of the subordinator
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the subordinator.
    /// * `duration` - The duration of the subordinator.
    /// * `time_step` - The time step of the subordinator.
    ///
    /// # Returns
    ///
    /// A f64 representing the occupation time of the subordinator.
    ///
    /// # Example
    ///
    /// ```rust
    /// let subordinator = Subordinator::new(0.5).unwrap();
    /// let ot = subordinator.occupation_time((-1.0, 1.0), 10.0, 0.1).unwrap();
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

/// impl `ContinuousProcess` trait for Levy process
impl ContinuousProcess for Levy {
    /// Simulate subordinator
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the subordinator simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the subordinator simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let subordinator = Subordinator::new(0.5).unwrap();
    /// let params = 0.1;
    /// let (t, x) = subordinator.simulate(params).unwrap();
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
        simulate_subordinator(self.alpha, duration, time_step)
    }
}

/// Simulate subordinator
///
/// This function simulates subordinator.
///
/// # Arguments
///
/// * `alpha` - The stability index of the subordinator.
/// * `duration` - The duration of the subordinator.
/// * `time_step` - The time step of the subordinator.
///
/// # Returns
///
/// The result of the subordinator simulation.   
///
/// # Example
///
/// ```rust
/// let subordinator = Subordinator::new(0.5).unwrap();
/// let params = 0.1;
/// let (t, x) = subordinator.simulate(params).unwrap();
/// ```
pub fn simulate_subordinator(
    alpha: f64,
    duration: impl Into<f64>,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let duration = duration.into();
    let num_steps = (duration / time_step).ceil() as usize;
    let t = (0..=num_steps)
        .into_par_iter()
        .map(|i| time_step * i as f64)
        .collect::<Vec<_>>();
    let noise = stable::skew_rands(alpha, num_steps)?
        .into_par_iter()
        .map(|x| x * time_step.powf(1.0 / alpha))
        .collect::<Vec<_>>();
    let x = cumsum(0.0, &noise);
    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_subordinator() {
        let subordinator = Subordinator::new(0.5).unwrap();
        let time_step = 0.1;
        let duration = 1.0;
        let (t, x) = subordinator.simulate(duration, time_step).unwrap();
        println!("t: {:?}", t);
        println!("x: {:?}", x);
    }

    #[test]
    fn test_fpt() {
        let subordinator = Subordinator::new(0.5).unwrap();
        let time_step = 0.1;
        let fpt = subordinator.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {:?}", fpt);
    }

    #[test]
    fn test_occupation_time() {
        let subordinator = Subordinator::new(0.5).unwrap();
        let time_step = 0.1;
        let ot = subordinator.occupation_time((-1.0, 1.0), 10.0, time_step).unwrap();
        println!("ot: {:?}", ot);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Subordinator>();
    }
}
