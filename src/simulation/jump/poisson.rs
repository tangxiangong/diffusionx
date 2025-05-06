//! Poisson process simulation

use crate::{SimulationError, XResult, random::exponential, simulation::prelude::*, utils::cumsum};

/// Poisson process
///
/// This struct represents a Poisson process.
///
/// # Fields
///
/// * `lambda` - The rate of the Poisson process.
#[derive(Debug, Clone)]
pub struct Poisson {
    lambda: f64,
}

impl Poisson {
    /// Create a new Poisson process simulation.
    ///
    /// # Arguments
    ///
    /// * `lambda` - The rate of the Poisson process.
    pub fn new(lambda: impl Into<f64>) -> XResult<Self> {
        let lambda = lambda.into();
        if lambda <= 0.0 {
            return Err(SimulationError::InvalidParameters(
                "lambda must be greater than 0".to_string(),
            )
            .into());
        }
        Ok(Self { lambda })
    }

    /// Get the rate of the Poisson process.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl PointProcess for Poisson {
    fn simulate_with_step(&self, num_step: usize) -> XResult<Pair> {
        simulate_poisson_with_step(self.lambda, num_step)
    }
}

/// Simulate the Poisson process with a given number of steps
///
/// # Arguments
///
/// * `lambda` - The rate of the Poisson process.
/// * `num_step` - The number of steps of the simulation.
pub fn simulate_poisson_with_step(lambda: impl Into<f64>, num_step: usize) -> XResult<Pair> {
    let lambda = lambda.into();
    let durations = exponential::rands(lambda, num_step)?;
    let t = cumsum(0.0, &durations);
    let x = (0..=num_step).map(|i| i as f64).collect::<Vec<_>>();
    Ok((t, x))
}

/// Simulate the Poisson process with a given number of steps
///
/// # Arguments
///
/// * `lambda` - The rate of the Poisson process.
/// * `duration` - The duration of the simulation.
pub fn simulate_poisson_with_duration(
    lambda: impl Into<f64>,
    duration: impl Into<f64>,
) -> XResult<Pair> {
    let poisson = Poisson::new(lambda)?;
    poisson.simulate_with_duration(duration)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fpt() {
        let poisson = Poisson::new(1.0).unwrap();
        let fpt = poisson.fpt((0.0, 1.0), 100.0).unwrap();
        assert!(fpt.is_some());
    }

    #[test]
    fn test_occupation_time() {
        let poisson = Poisson::new(1.0).unwrap();
        let ot = poisson.occupation_time((0.0, 1.0), 100.0).unwrap();
        assert!(ot > 0.0);
    }

    #[test]
    fn test_raw_moment() {
        let poisson = Poisson::new(1.0).unwrap();
        let moment = poisson.raw_moment(100.0, 1, 100).unwrap();
        assert!(moment > 0.0);
    }

    #[test]
    fn test_central_moment() {
        let poisson = Poisson::new(1.0).unwrap();
        let _moment = poisson.central_moment(100.0, 1, 100).unwrap();
        // assert!(moment > 0.0);
    }

    #[test]
    fn test_simulate_with_step() {
        let poisson = Poisson::new(1.0).unwrap();
        let (t, x) = poisson.simulate_with_step(100).unwrap();
        assert!(t.len() == 101);
        assert!(x.len() == 101);
    }

    #[test]
    fn test_simulate_with_duration() {
        let poisson = Poisson::new(1.0).unwrap();
        let (t, _) = poisson.simulate_with_duration(100.0).unwrap();
        assert!(*t.last().unwrap() <= 100.0);
    }
}
