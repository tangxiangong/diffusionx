//! Birth-death process simulation

use crate::{
    SimulationError, XResult,
    random::{exponential, uniform},
    simulation::prelude::*,
    utils::cumsum,
};
use rayon::prelude::*;

/// Birth-death process
///
/// This struct represents a Birth-death process.
///
/// # Mathematical Formulation
///
/// The Birth-death process is a process that describes the number of particles in a system that can either birth or die.
///
/// # Fields
///
/// * `lambda` - The rate of birth.
/// * `mu` - The rate of death.
#[derive(Debug, Clone)]
pub struct BirthDeath {
    lambda: f64,
    mu: f64,
}

impl BirthDeath {
    /// Create a new Birth-death process simulation.
    ///
    /// # Arguments
    ///
    /// * `lambda` - The rate of birth.
    /// * `mu` - The rate of death.
    pub fn new(lambda: impl Into<f64>, mu: impl Into<f64>) -> XResult<Self> {
        let lambda = lambda.into();
        let mu = mu.into();
        if lambda <= 0.0 {
            return Err(SimulationError::InvalidParameters(
                "lambda must be greater than 0".to_string(),
            )
            .into());
        }
        if mu <= 0.0 {
            return Err(SimulationError::InvalidParameters(
                "mu must be greater than 0".to_string(),
            )
            .into());
        }
        Ok(Self { lambda, mu })
    }

    /// Get the rate of birth.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Get the rate of death.
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Simulate the Birth-death process with a given number of steps
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
    /// let birth_death = BirthDeath::new(1.0, 1.0);
    /// let (t, x) = birth_death.simulate_with_step(100).unwrap();
    /// ```
    pub fn simulate_with_step(&self, num_step: usize) -> XResult<PointPair> {
        let traj = self.step(num_step)?;
        traj.simulate_with_step()
    }

    /// Simulate the Birth-death process with a given duration
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
    /// let birth_death = BirthDeath::new(1.0, 1.0);
    /// let (t, x) = birth_death.simulate_with_duration(100.0).unwrap();
    /// ```
    pub fn simulate_with_duration(&self, duration: impl Into<f64>) -> XResult<PointPair> {
        let traj = self.duration(duration)?;
        traj.simulate_with_duration()
    }

    /// Get the raw moment of the Birth-death process
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
    /// let birth_death = BirthDeath::new(1.0, 1.0);
    /// let moment = birth_death.raw_moment(100.0, 1, 100).unwrap();
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

    /// Get the central moment of the Birth-death process   
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
    /// let birth_death = BirthDeath::new(1.0, 1.0);
    /// let moment = birth_death.central_moment(100.0, 1, 100).unwrap();
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

    /// Get the first passage time of the Birth-death process
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the simulation.
    /// * `max_duration` - The maximum duration of the simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the first passage time of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let birth_death = BirthDeath::new(1.0, 1.0);
    /// let fpt = birth_death.fpt((0.0, 1.0), 100.0).unwrap();
    /// ```
    pub fn fpt(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        max_duration: impl Into<f64>,
    ) -> XResult<Option<f64>> {
        let fpt = FirstPassageTime::new(self, domain)?;
        fpt.simulate_p(max_duration)
    }

    /// Get the occupation time of the Birth-death process
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the simulation.
    /// * `duration` - The duration of the simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the occupation time of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let birth_death = BirthDeath::new(1.0, 1.0);
    /// let ot = birth_death.occupation_time((0.0, 1.0), 100.0).unwrap();
    /// ```
    pub fn occupation_time(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        duration: impl Into<f64>,
    ) -> XResult<f64> {
        let ot = OccupationTime::new(self, domain, duration)?;
        ot.simulate_p()
    }
}

impl PointProcess for BirthDeath {
    fn simulate_with_step(&self, num_step: usize) -> XResult<PointPair> {
        simulate_birth_death_with_step(self.lambda, self.mu, num_step)
    }
}

/// Simulate the Birth-death process with a given number of steps
///
/// # Arguments
///
/// * `lambda` - The rate of birth.
/// * `mu` - The rate of death.
/// * `num_step` - The number of steps of the simulation.
pub fn simulate_birth_death_with_step(
    lambda: impl Into<f64>,
    mu: impl Into<f64>,
    num_step: usize,
) -> XResult<PointPair> {
    let lambda = lambda.into();
    let mu = mu.into();
    let durations = exponential::rands(lambda + mu, num_step)?;
    let t = cumsum(0.0, &durations);
    let directions = uniform::bool_rands(lambda / (lambda + mu), num_step)?
        .into_par_iter()
        .map(|b| if b { 1i64 } else { -1i64 })
        .collect::<Vec<_>>();
    let x = cumsum(0i64, &directions);
    Ok((t, x))
}

/// Simulate the Birth-death process with a given duration
///
/// # Arguments
///
/// * `lambda` - The rate of birth.
/// * `mu` - The rate of death.
/// * `duration` - The duration of the simulation.
pub fn simulate_birth_death_with_duration(
    lambda: impl Into<f64>,
    mu: impl Into<f64>,
    duration: impl Into<f64>,
) -> XResult<PointPair> {
    let birth_death = BirthDeath::new(lambda, mu)?;
    birth_death.simulate_with_duration(duration)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fpt() {
        let birth_death = BirthDeath::new(1.0, 1.0).unwrap();
        let fpt = birth_death.fpt((0.0, 1.0), 100.0).unwrap();
        assert!(fpt.is_some());
    }

    #[test]
    fn test_occupation_time() {
        let birth_death = BirthDeath::new(1.0, 1.0).unwrap();
        let ot = birth_death.occupation_time((0.0, 1.0), 100.0).unwrap();
        assert!(ot > 0.0);
    }

    #[test]
    fn test_raw_moment() {
        let birth_death = BirthDeath::new(1.0, 1.0).unwrap();
        let _moment = birth_death.raw_moment(100.0, 1, 100).unwrap();
        // assert!(moment > 0.0);
    }

    #[test]
    fn test_central_moment() {
        let birth_death = BirthDeath::new(1.0, 1.0).unwrap();
        let _moment = birth_death.central_moment(100.0, 1, 100).unwrap();
        // assert!(moment > 0.0);
    }

    #[test]
    fn test_simulate_with_step() {
        let birth_death = BirthDeath::new(1.0, 1.0).unwrap();
        let (t, x) = birth_death.simulate_with_step(100).unwrap();
        assert!(t.len() == 101);
        assert!(x.len() == 101);
    }

    #[test]
    fn test_simulate_with_duration() {
        let birth_death = BirthDeath::new(1.0, 1.0).unwrap();
        let (t, _) = birth_death.simulate_with_duration(100.0).unwrap();
        assert!(*t.last().unwrap() <= 100.0);
    }
}
