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

    /// Simulate the Poisson process with a given number of steps
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let poisson = Poisson::new(1.0);
    /// let (t, x) = poisson.simulate_with_step(100).unwrap();
    /// ```
    pub fn simulate_with_step(&self, num_step: usize) -> XResult<PointPair> {
        let traj = self.step(num_step)?;
        traj.simulate_with_step()
    }

    /// Simulate the Poisson process with a given duration
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let poisson = Poisson::new(1.0);
    /// let (t, x) = poisson.simulate_with_duration(100.0).unwrap();
    /// ```
    pub fn simulate_with_duration(&self, duration: impl Into<f64>) -> XResult<PointPair> {
        let traj = self.duration(duration)?;
        traj.simulate_with_duration()
    }

    /// Get the raw moment of the Poisson process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    ///
    /// # Returns
    ///
    /// A f64 representing the raw moment of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let poisson = Poisson::new(1.0);
    /// let moment = poisson.raw_moment(100.0, 1, 100).unwrap();
    /// ```
    pub fn raw_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
    ) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.raw_moment(order, particles, 0.1)
    }

    /// Get the central moment of the Poisson process   
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    ///
    /// # Returns
    ///
    /// A f64 representing the central moment of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let poisson = Poisson::new(1.0);
    /// let moment = poisson.central_moment(100.0, 1, 100).unwrap();
    /// ```
    pub fn central_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
    ) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.central_moment(order, particles, 0.1)
    }

    pub fn fpt(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        max_duration: impl Into<f64>,
    ) -> XResult<Option<f64>> {
        let fpt = FirstPassageTime::new(self, domain)?;
        fpt.simulate_p(max_duration)
    }

    pub fn occupation_time(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        duration: impl Into<f64>,
    ) -> XResult<f64> {
        let ot = OccupationTime::new(self, domain, duration)?;
        ot.simulate_p()
    }
}

impl PointProcess for Poisson {
    fn simulate_with_step(&self, num_step: usize) -> XResult<PointPair> {
        simulate_poisson_with_step(self.lambda, num_step)
    }
}

/// Simulate the Poisson process with a given number of steps
///
/// # Arguments
///
/// * `lambda` - The rate of the Poisson process.
/// * `num_step` - The number of steps of the simulation.
pub fn simulate_poisson_with_step(lambda: impl Into<f64>, num_step: usize) -> XResult<PointPair> {
    let lambda = lambda.into();
    let durations = exponential::rands(lambda, num_step)?;
    let t = cumsum(0.0, &durations);
    let x = (0..=num_step as i64).collect::<Vec<_>>();
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
) -> XResult<PointPair> {
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
