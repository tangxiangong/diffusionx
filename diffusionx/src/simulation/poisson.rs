//! Poisson process simulation


use crate::{
    SimulationError, XResult,
    random::exponential,
    simulation::prelude::*,
    utils::cumsum,
};

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
    fn simulate_with_duration(&self, duration: impl Into<f64>) -> XResult<PointPair> {
        let duration = duration.into();
        let mut num_step = (duration / self.lambda).ceil() as usize;
        let (t, x) = loop {
            let (t, x) = self.simulate_with_step(num_step)?;
            if t.last().is_none() {
                return Err(SimulationError::Unknown.into());
            }
            let end_time = *t.last().unwrap();
            if end_time >= duration {
                break (t, x);
            }
            num_step *= 2;
        };
        let index = t.iter().position(|&time| time >= duration).unwrap();
        let mut t_ = vec![0.0; index+1];
        let mut x_ = vec![0i64; index+1];
        for i in 0..=index-1 {
            t_[i] = t[i];
            x_[i] = x[i];
        }
        if t[index] > duration {
            t_[index] = duration;
            x_[index] = x_[index-1];
        } else {
            t_[index] = t[index];
            x_[index] = x[index];
        }

        Ok((t_, x_))
    }           
}
