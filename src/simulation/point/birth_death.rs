//! Birth-death process simulation

use crate::{
    RealExt, SimulationError, XResult,
    random::{PAR_THRESHOLD, exponential},
    simulation::prelude::*,
    utils::cumsum,
};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Exp1};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// Birth-death process
///
/// # Mathematical Formulation
///
/// The Birth-death process is a process that describes the number of particles in a system that can either birth or die.
#[derive(Debug, Clone)]
pub struct BirthDeath<T: FloatExt = f64, X: RealExt = T> {
    /// The rate of birth
    lambda: T,
    /// The rate of death
    mu: T,
    _marker: std::marker::PhantomData<X>,
}

impl<T: FloatExt, X: RealExt> BirthDeath<T, X> {
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
    pub fn new(lambda: T, mu: T) -> XResult<Self> {
        if lambda <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `lambda` must be greater than 0, got {lambda:?}"
            ))
            .into());
        }
        if mu <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `mu` must be greater than 0, got {mu:?}"
            ))
            .into());
        }
        Ok(Self {
            lambda,
            mu,
            _marker: std::marker::PhantomData,
        })
    }

    /// Get the rate of birth
    pub fn get_lambda(&self) -> T {
        self.lambda
    }

    /// Get the rate of death
    pub fn get_mu(&self) -> T {
        self.mu
    }
}

impl<T: FloatExt, X: RealExt + std::ops::Neg<Output = X>> PointProcess<T, X> for BirthDeath<T, X>
where
    Exp1: Distribution<T>,
{
    fn start(&self) -> X {
        X::zero()
    }

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
    fn simulate_with_step(&self, num_step: usize) -> XResult<(Vec<T>, Vec<X>)> {
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
pub fn simulate_birth_death_with_step<T: FloatExt, X: RealExt + std::ops::Neg<Output = X>>(
    lambda: T,
    mu: T,
    num_step: usize,
) -> XResult<(Vec<T>, Vec<X>)>
where
    Exp1: Distribution<T>,
{
    if lambda <= T::zero() {
        return Err(SimulationError::InvalidParameters(format!(
            "The `lambda` must be greater than 0, got {lambda:?}"
        ))
        .into());
    }
    if mu <= T::zero() {
        return Err(SimulationError::InvalidParameters(format!(
            "The `mu` must be greater than 0, got {mu:?}"
        ))
        .into());
    }
    if num_step == 0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `num_step` must be greater than 0, got {num_step}"
        ))
        .into());
    }
    let durations = exponential::rands(lambda + mu, num_step)?;
    let prob = (lambda / (lambda + mu)).to_f64().unwrap();
    let t = cumsum(T::zero(), &durations);
    let directions = if num_step <= PAR_THRESHOLD {
        let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
        (0..num_step)
            .map(|_| {
                let dir = rng.random_bool(prob);
                if dir { X::one() } else { -X::one() }
            })
            .collect::<Vec<_>>()
    } else {
        (0..num_step)
            .into_par_iter()
            .map_init(
                || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                |r, _| r.random_bool(prob),
            )
            .map(|b| if b { X::one() } else { -X::one() })
            .collect::<Vec<_>>()
    };
    let x = cumsum(X::zero(), &directions);
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
pub fn simulate_birth_death_with_duration<T: FloatExt, X: RealExt + std::ops::Neg<Output = X>>(
    lambda: T,
    mu: T,
    duration: T,
) -> XResult<(Vec<T>, Vec<X>)>
where
    Exp1: Distribution<T>,
{
    if lambda <= T::zero() {
        return Err(SimulationError::InvalidParameters(format!(
            "The `lambda` must be greater than 0, got {lambda:?}"
        ))
        .into());
    }
    if mu <= T::zero() {
        return Err(SimulationError::InvalidParameters(format!(
            "The `mu` must be greater than 0, got {mu:?}"
        ))
        .into());
    }
    if duration <= T::zero() {
        return Err(SimulationError::InvalidParameters(format!(
            "The `duration` must be positive, got `{duration:?}`"
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
        let birth_death: BirthDeath<_, i32> = BirthDeath::new(1.0, 1.0).unwrap();
        let _moment = birth_death.raw_moment(100.0, 1, 100).unwrap();
        // assert!(moment > 0.0);
    }

    #[test]
    fn test_central_moment() {
        let birth_death: BirthDeath<_, i32> = BirthDeath::new(1.0, 1.0).unwrap();
        let _moment = birth_death.central_moment(100.0, 1, 100).unwrap();
        // assert!(moment > 0.0);
    }

    #[test]
    fn test_simulate_with_step() {
        let birth_death: BirthDeath<_, i32> = BirthDeath::new(1.0, 1.0).unwrap();
        let (t, x) = birth_death.simulate_with_step(100).unwrap();
        assert!(t.len() == 101);
        assert!(x.len() == 101);
    }

    #[test]
    fn test_simulate_with_duration() {
        let birth_death: BirthDeath<_, i32> = BirthDeath::new(1.0, 1.0).unwrap();
        let (t, _) = birth_death.simulate_with_duration(100.0).unwrap();
        assert!(*t.last().unwrap() <= 100.0);
    }
}
