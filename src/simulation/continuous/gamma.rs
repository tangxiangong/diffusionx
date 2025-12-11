//! Gamma process simulation

use crate::{
    FloatExt, SimulationError, XError, XResult, check_duration_time_step, random::gamma,
    simulation::prelude::*,
};
use rand::prelude::*;
use rand_distr::{Distribution, Exp1, Open01, StandardNormal};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// Gamma process
///
/// # Mathematical Formulation
///
/// A Gamma process is a process that is non-negative and has a non-decreasing sample path with a Gamma distribution.
#[derive(Debug, Clone)]
pub struct Gamma<T: FloatExt = f64> {
    /// The shape parameter
    shape: T,
    /// The rate parameter
    rate: T,
}

impl<T: FloatExt> Gamma<T> {
    /// Create a new `Gamma`
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape parameter.
    /// * `rate` - The rate parameter.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::Gamma;
    ///
    /// let gamma = Gamma::new(0.5, 1.0).unwrap();
    /// ```
    pub fn new(shape: T, rate: T) -> XResult<Self> {
        if shape <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `shape` must be positive, got {shape:?}"
            ))
            .into());
        }
        if rate <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `rate` must be positive, got {rate:?}"
            ))
            .into());
        }
        Ok(Self { shape, rate })
    }

    /// Get the shape parameter
    pub fn get_shape(&self) -> T {
        self.shape
    }

    /// Get the rate parameter
    pub fn get_rate(&self) -> T {
        self.rate
    }
}

impl<T: FloatExt> ContinuousProcess<T> for Gamma<T>
where
    StandardNormal: Distribution<T>,
    Exp1: Distribution<T>,
    Open01: Distribution<T>,
{
    fn start(&self) -> T {
        T::zero()
    }

    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        simulate_gamma(self.shape, self.rate, duration, time_step)
    }

    fn displacement(&self, duration: T, time_step: T) -> XResult<T> {
        check_duration_time_step(duration, time_step)?;

        let num_steps = (duration / time_step).ceil().to_usize().unwrap();
        let scale = T::one() / self.rate;
        let gamma = rand_distr::Gamma::new(self.shape, scale)
            .map_err(|e| XError::InvalidParameters(e.to_string()))?;
        let mut delta_x = (0..num_steps - 1)
            .into_par_iter()
            .map_init(
                || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                |r, _| r.sample(gamma),
            )
            .sum();

        let last_step = duration - (T::from(num_steps - 1).unwrap() * time_step);
        delta_x += gamma::rand(self.shape * last_step, scale)?;

        Ok(delta_x)
    }
}

/// Simulate Gamma process
///
/// # Arguments
///
/// * `shape` - The shape parameter.
/// * `rate` - The rate parameter.
/// * `duration` - The duration of the trajectory.
/// * `time_step` - The time step of the simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::gamma::simulate_gamma;
///
/// let shape = 0.5;
/// let rate = 1.0;
/// let duration = 1.0;
/// let time_step = 0.1;
/// let (t, x) = simulate_gamma(shape, rate, duration, time_step).unwrap();
/// ```
pub fn simulate_gamma<T: FloatExt>(
    shape: T,
    rate: T,
    duration: T,
    time_step: T,
) -> XResult<(Vec<T>, Vec<T>)>
where
    StandardNormal: Distribution<T>,
    Exp1: Distribution<T>,
    Open01: Distribution<T>,
{
    check_duration_time_step(duration, time_step)?;

    let num_steps = (duration / time_step).ceil().to_usize().unwrap();

    let scale = T::one() / rate;
    let noise = gamma::rands(shape * time_step, scale, num_steps - 1)?;

    let mut t = Vec::with_capacity(num_steps + 1);
    let mut x = Vec::with_capacity(num_steps + 1);

    t.push(T::zero());
    x.push(T::zero());

    let mut current_x = T::zero();
    let mut current_t = T::zero();
    for xi in noise {
        current_t += time_step;
        t.push(current_t);
        current_x += xi;
        x.push(current_x);
    }

    let last_step = duration - current_t;
    let xi = gamma::rand(shape * last_step, scale)?;
    current_x += xi;
    x.push(current_x);
    t.push(duration);

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
        println!("t: {t:?}");
        println!("x: {x:?}");
    }

    #[test]
    fn test_fpt() {
        let gamma = Gamma::new(0.5, 1.0).unwrap();
        let time_step = 0.1;
        let fpt = gamma.fpt((-1.0, 1.0), 1000.0, time_step).unwrap().unwrap();
        println!("fpt: {fpt:?}");
    }

    #[test]
    fn test_occupation_time() {
        let gamma = Gamma::new(0.5, 1.0).unwrap();
        let time_step = 0.1;
        let ot = gamma.occupation_time((-1.0, 1.0), 10.0, time_step).unwrap();
        println!("ot: {ot:?}");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Gamma>();
    }
}
