//! Poisson process simulation

use rand_distr::{Distribution, Exp1};

use crate::{
    RealExt, SimulationError, XResult, random::exponential, simulation::prelude::*, utils::cumsum,
};

/// Poisson process
#[derive(Debug, Clone)]
pub struct Poisson<T: FloatExt, X: RealExt = T> {
    /// The rate of the Poisson process
    lambda: T,
    _marker: std::marker::PhantomData<X>,
}

impl<T: FloatExt, X: RealExt> Poisson<T, X> {
    /// Create a new `Poisson`
    ///
    /// # Arguments
    ///
    /// * `lambda` - The rate of the Poisson process.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::point::Poisson;
    ///
    /// let poisson = Poisson::new(1.0).unwrap();
    /// ```
    pub fn new(lambda: T) -> XResult<Self> {
        if lambda <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `lambda` must be greater than 0, but got {lambda:?}"
            ))
            .into());
        }
        Ok(Self {
            lambda,
            _marker: std::marker::PhantomData,
        })
    }

    /// Get the rate
    pub fn get_lambda(&self) -> T {
        self.lambda
    }
}

impl<T: FloatExt, X: RealExt> PointProcess<T, X> for Poisson<T, X>
where
    Exp1: Distribution<T>,
{
    fn start(&self) -> X {
        X::zero()
    }

    /// Simulate the Poisson process with a given number of steps
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{point::Poisson, prelude::*};
    ///
    /// let poisson = Poisson::new(1.0).unwrap();
    /// let (t, x) = poisson.simulate_with_step(1000).unwrap();
    /// ```
    fn simulate_with_step(&self, num_step: usize) -> XResult<(Vec<T>, Vec<X>)> {
        simulate_poisson_with_step(self.lambda, num_step)
    }
}

/// Simulate the Poisson process with a given number of steps
///
/// # Arguments
///
/// * `lambda` - The rate of the Poisson process.
/// * `num_step` - The number of steps of the simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::point::poisson::simulate_poisson_with_step;
///
/// let (t, x) = simulate_poisson_with_step(1.0, 1000).unwrap();
/// ```
pub fn simulate_poisson_with_step<T: FloatExt, X: RealExt>(
    lambda: T,
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
    let durations = exponential::rands(lambda, num_step)?;
    let t = cumsum(T::zero(), &durations);
    let x = (0..=num_step)
        .map(|i| X::from(i).unwrap())
        .collect::<Vec<_>>();
    Ok((t, x))
}

/// Simulate the Poisson process with a given duration
///
/// # Arguments
///
/// * `lambda` - The rate of the Poisson process.
/// * `duration` - The duration of the simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::point::poisson::simulate_poisson_with_duration;
///
/// let (t, x) = simulate_poisson_with_duration(1.0, 100.0).unwrap();
/// ```
pub fn simulate_poisson_with_duration(lambda: f64, duration: f64) -> XResult<(Vec<f64>, Vec<f64>)> {
    if lambda <= 0.0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `lambda` must be greater than 0, got {lambda}"
        ))
        .into());
    }
    if duration <= 0.0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `duration` must be positive, got `{duration}`"
        ))
        .into());
    }
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
        let poisson: Poisson<f64, u32> = Poisson::new(1.0).unwrap();
        let moment = poisson.raw_moment(100.0, 1, 100).unwrap();
        assert!(moment > 0.0);
    }

    #[test]
    fn test_central_moment() {
        let poisson: Poisson<f64, u32> = Poisson::new(1.0).unwrap();
        let _moment = poisson.central_moment(100.0, 1, 100).unwrap();
        // assert!(moment > 0.0);
    }

    #[test]
    fn test_simulate_with_step() {
        let poisson: Poisson<f64, u32> = Poisson::new(1.0).unwrap();
        let (t, x) = poisson.simulate_with_step(100).unwrap();
        assert!(t.len() == 101);
        assert!(x.len() == 101);
    }

    #[test]
    fn test_simulate_with_duration() {
        let poisson: Poisson<f64, u32> = Poisson::new(1.0).unwrap();
        let (t, _) = poisson.simulate_with_duration(100.0).unwrap();
        assert!(*t.last().unwrap() <= 100.0);
    }
}
