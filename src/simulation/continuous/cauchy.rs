//! Cauchy process
//!
//! The Cauchy process is a Lévy process with alpha = 1.

use num_traits::FloatConst;
use rand_distr::{Distribution, Exp1, uniform::SampleUniform};

use crate::{
    FloatExt, SimulationError, XResult,
    simulation::{
        continuous::{AsymmetricLevy, Levy, simulate_asymmetric_levy, simulate_levy},
        prelude::*,
    },
};

/// Asymmetric Cauchy process
#[derive(Debug, Clone)]
pub struct AsymmetricCauchy<T: FloatExt = f64> {
    /// The starting position
    start_position: T,
    /// The asymmetry parameter
    beta: T,
}

impl<T: FloatExt> AsymmetricCauchy<T> {
    /// Create a new `AsymmetricCauchy`
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position.
    /// * `beta` - The asymmetry parameter.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::AsymmetricCauchy;
    ///
    /// let cauchy = AsymmetricCauchy::new(0.0, 0.4).unwrap();
    /// ```
    pub fn new(start_position: T, beta: T) -> XResult<Self> {
        if !(-T::one()..=T::one()).contains(&beta) {
            return Err(SimulationError::InvalidParameters(format!(
                "The `beta` must be in the range [-1, 1], got {beta:?}",
            ))
            .into());
        }
        Ok(Self {
            start_position,
            beta,
        })
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> T {
        self.start_position
    }

    /// Get the asymmetry parameter
    pub fn get_asymmetry(&self) -> T {
        self.beta
    }
}

impl<T: FloatExt + FloatConst + SampleUniform> ContinuousProcess<T> for AsymmetricCauchy<T>
where
    Exp1: Distribution<T>,
{
    fn start(&self) -> T {
        self.start_position
    }

    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        simulate_asymmetric_levy(
            self.start_position,
            T::one(),
            self.beta,
            duration,
            time_step,
        )
    }

    fn displacement(&self, duration: T, time_step: T) -> XResult<T> {
        let sp = AsymmetricLevy::new(self.start_position, T::one(), self.beta)?;
        sp.displacement(duration, time_step)
    }
}

/// Cauchy process
#[derive(Debug, Clone)]
pub struct Cauchy<T: FloatExt = f64> {
    /// The starting position
    start_position: T,
}

impl<T: FloatExt> Default for Cauchy<T> {
    fn default() -> Self {
        Self {
            start_position: T::zero(),
        }
    }
}

impl<T: FloatExt> Cauchy<T> {
    /// Create a new `Cauchy`
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::Cauchy;
    ///
    /// let cauchy = Cauchy::new(0.0).unwrap();
    /// ```
    pub fn new(start_position: T) -> Self {
        Self { start_position }
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> T {
        self.start_position
    }
}

impl<T: FloatExt + FloatConst + SampleUniform> ContinuousProcess<T> for Cauchy<T>
where
    Exp1: Distribution<T>,
{
    fn start(&self) -> T {
        self.start_position
    }

    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        simulate_levy(self.start_position, T::one(), duration, time_step)
    }

    fn displacement(&self, duration: T, time_step: T) -> XResult<T> {
        let sp = Levy::new(self.start_position, T::one())?;
        sp.displacement(duration, time_step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_cauchy() {
        let cauchy = Cauchy::default();
        let asymmetric_cauchy = AsymmetricCauchy::new(0.0, 0.4).unwrap();
        let time_step = 0.1;
        let duration = 1.0;
        let (t, x) = cauchy.simulate(duration, time_step).unwrap();
        println!("t: {t:?}");
        println!("x: {x:?}");
        let (t, x) = asymmetric_cauchy.simulate(duration, time_step).unwrap();
        println!("t: {t:?}");
        println!("x: {x:?}");
    }

    #[test]
    fn test_fpt() {
        let cauchy = Cauchy::default();
        let asymmetric_cauchy = AsymmetricCauchy::new(0.0, 0.4).unwrap();
        let time_step = 0.1;
        let fpt = cauchy.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {fpt:?}");
        let fpt = asymmetric_cauchy
            .fpt((-1.0, 1.0), 1000.0, time_step)
            .unwrap();
        println!("fpt: {fpt:?}");
    }

    #[test]
    fn test_occupation_time() {
        let cauchy = Cauchy::default();
        let asymmetric_cauchy = AsymmetricCauchy::new(0.0, 0.4).unwrap();
        let time_step = 0.1;
        let ot = cauchy
            .occupation_time((-1.0, 1.0), 10.0, time_step)
            .unwrap();
        println!("ot: {ot:?}");
        let ot = asymmetric_cauchy
            .occupation_time((-1.0, 1.0), 10.0, time_step)
            .unwrap();
        println!("ot: {ot:?}");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Cauchy>();
        assert_send_sync::<AsymmetricCauchy>();
    }
}
