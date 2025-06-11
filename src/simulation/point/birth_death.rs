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
/// # Mathematical Formulation
///
/// The Birth-death process is a process that describes the number of particles in a system that can either birth or die.
#[derive(Debug, Clone)]
pub struct BirthDeath {
    /// The rate of birth
    lambda: f64,
    /// The rate of death
    mu: f64,
}

impl BirthDeath {
    /// Create a new `BirthDeath`
    ///
    /// # Arguments
    ///
    /// * `lambda` - The rate of birth.
    /// * `mu` - The rate of death.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::point::BirthDeath;
    ///
    /// let birth_death = BirthDeath::new(1.0, 1.0).unwrap();
    /// ```
    pub fn new(lambda: impl Into<f64>, mu: impl Into<f64>) -> XResult<Self> {
        let lambda = lambda.into();
        let mu = mu.into();
        if lambda <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `lambda` must be greater than 0, got {}",
                lambda
            ))
            .into());
        }
        if mu <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `mu` must be greater than 0, got {}",
                mu
            ))
            .into());
        }
        Ok(Self { lambda, mu })
    }

    /// Get the rate of birth
    pub fn get_lambda(&self) -> f64 {
        self.lambda
    }

    /// Get the rate of death
    pub fn get_mu(&self) -> f64 {
        self.mu
    }
}

impl PointProcess for BirthDeath {
    /// Simulate the Birth-death process with a given number of steps
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{point::BirthDeath, prelude::*};
    ///
    /// let birth_death = BirthDeath::new(1.0, 1.0).unwrap();
    /// let (t, x) = birth_death.simulate_with_step(100).unwrap();
    /// ```
    fn simulate_with_step(&self, num_step: usize) -> XResult<Pair> {
        simulate_birth_death_with_step(self.lambda, self.mu, num_step)
    }
}

/// Simulate the Birth-death process with a given number of steps
///
/// # Arguments
///
/// * `lambda` - The rate of birth.
/// * `mu` - The rate of death.
/// * `num_step` - The number of steps
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::jump::birth_death::simulate_birth_death_with_step;
///
/// let (t, x) = simulate_birth_death_with_step(1.0, 1.0, 100).unwrap();
/// ```
pub fn simulate_birth_death_with_step(lambda: f64, mu: f64, num_step: usize) -> XResult<Pair> {
    if lambda <= 0.0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `lambda` must be greater than 0, got {}",
            lambda
        ))
        .into());
    }
    if mu <= 0.0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `mu` must be greater than 0, got {}",
            mu
        ))
        .into());
    }
    if num_step == 0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `num_step` must be greater than 0, got {}",
            num_step
        ))
        .into());
    }
    let durations = exponential::rands(lambda + mu, num_step)?;
    let t = cumsum(0.0, &durations);
    let directions = uniform::bool_rands(lambda / (lambda + mu), num_step)?
        .into_par_iter()
        .map(|b| if b { 1.0 } else { -1.0 })
        .collect::<Vec<_>>();
    let x = cumsum(0.0, &directions);
    Ok((t, x))
}

/// Simulate the Birth-death process with a given duration
///
/// # Arguments
///
/// * `lambda` - The rate of birth.
/// * `mu` - The rate of death.
/// * `duration` - The duration of the simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::point::birth_death::simulate_birth_death_with_duration;
///
/// let (t, x) = simulate_birth_death_with_duration(1.0, 1.0, 100.0).unwrap();
/// ```
pub fn simulate_birth_death_with_duration(lambda: f64, mu: f64, duration: f64) -> XResult<Pair> {
    if lambda <= 0.0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `lambda` must be greater than 0, got {}",
            lambda
        ))
        .into());
    }
    if mu <= 0.0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `mu` must be greater than 0, got {}",
            mu
        ))
        .into());
    }
    if duration <= 0.0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `duration` must be positive, got `{}`",
            duration
        ))
        .into());
    }
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
