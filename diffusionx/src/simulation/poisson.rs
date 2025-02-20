//! Poisson process simulation

use crate::{SimulationError, XResult, random::exponential, simulation::prelude::*, utils::cumsum};

/// Poisson process simulation
///
/// This struct represents a Poisson process simulation.
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
    fn simulate_with_step(&self, num_step: usize) -> XResult<PointPair> {
        let durations = exponential::rands(self.lambda, num_step)?;
        let t = cumsum(0.0, &durations);
        let x = (0..=num_step as i64).collect::<Vec<_>>();
        Ok((t, x))
    }
}
