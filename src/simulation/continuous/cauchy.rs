//! Cauchy process
//!
//! The Cauchy process is a Lévy process with alpha = 1.

use crate::{
    SimulationError, XResult,
    simulation::{
        continuous::{AsymmetricLevy, Levy, simulate_asymmetric_levy, simulate_levy},
        prelude::*,
    },
};

/// Asymmetric Cauchy process
#[derive(Debug, Clone)]
pub struct AsymmetricCauchy {
    /// The starting position
    start_position: f64,
    /// The asymmetry parameter
    beta: f64,
}

impl AsymmetricCauchy {
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
    pub fn new(start_position: impl Into<f64>, beta: impl Into<f64>) -> XResult<Self> {
        let start_position = start_position.into();
        let beta = beta.into();
        if !(-1.0..=1.0).contains(&beta) {
            return Err(SimulationError::InvalidParameters(format!(
                "The `beta` must be in the range [-1, 1], got {beta}",
            ))
            .into());
        }
        Ok(Self {
            start_position,
            beta,
        })
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> f64 {
        self.start_position
    }

    /// Get the asymmetry parameter
    pub fn get_asymmetry(&self) -> f64 {
        self.beta
    }
}

impl ContinuousProcess for AsymmetricCauchy {
    fn start(&self) -> f64 {
        self.start_position
    }

    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_asymmetric_cauchy(self.start_position, self.beta, duration, time_step)
    }

    fn displacement(&self, duration: f64, time_step: f64) -> XResult<f64> {
        let sp = AsymmetricLevy::new(self.start_position, 1.0, self.beta)?;
        sp.displacement(duration, time_step)
    }
}

/// Simulate asymmetric Cauchy process
///
/// # Arguments
///
/// * `start_position` - The starting position.
/// * `beta` - The asymmetry parameter.
/// * `duration` - The duration of the trajectory.
/// * `time_step` - The time step of the simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::cauchy::simulate_asymmetric_cauchy;
///
/// let time_step = 0.1;
/// let duration = 1.0;
/// let (t, x) = simulate_asymmetric_cauchy(0.0, 0.4, duration, time_step).unwrap();
/// ```
pub fn simulate_asymmetric_cauchy(
    start_position: f64,
    beta: f64,
    duration: f64,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    simulate_asymmetric_levy(start_position, 1.0, beta, duration, time_step)
}

/// Cauchy process
#[derive(Debug, Clone)]
pub struct Cauchy {
    /// The starting position
    start_position: f64,
}

impl Default for Cauchy {
    fn default() -> Self {
        Self {
            start_position: 0.0,
        }
    }
}

impl Cauchy {
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
    pub fn new(start_position: impl Into<f64>) -> Self {
        let start_position = start_position.into();
        Self { start_position }
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> f64 {
        self.start_position
    }
}

impl ContinuousProcess for Cauchy {
    fn start(&self) -> f64 {
        self.start_position
    }

    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_cauchy(self.start_position, duration, time_step)
    }

    fn displacement(&self, duration: f64, time_step: f64) -> XResult<f64> {
        let sp = Levy::new(self.start_position, 1.0)?;
        sp.displacement(duration, time_step)
    }
}

/// Simulate Cauchy process
///
/// # Arguments
///
/// * `start_position` - The starting position.
/// * `duration` - The duration of the trajectory.
/// * `time_step` - The time step of the simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::cauchy::simulate_cauchy;
///
/// let time_step = 0.1;
/// let duration = 1.0;
/// let (t, x) = simulate_cauchy(0.0, duration, time_step).unwrap();
/// ```
pub fn simulate_cauchy(
    start_position: f64,
    duration: f64,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    simulate_levy(start_position, 1.0, duration, time_step)
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
