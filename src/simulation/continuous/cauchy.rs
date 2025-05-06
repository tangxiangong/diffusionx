//! Cauchy process
//!
//! The Cauchy process is a Lévy process with alpha = 1.
//!
//! For Lévy process, see [`crate::simulation::continuous::levy`].

use crate::{SimulationError, XResult, simulation::prelude::*};

use super::{simulate_asymmetric_levy, simulate_levy};

/// Asymmetric Cauchy process
///
/// This struct represents a Cauchy process.
///
/// # Fields
///
/// * `start_position` - The starting position of the Cauchy process.
/// * `beta` - The asymmetry parameter of the asymmetric Cauchy process.
#[derive(Debug, Clone)]
pub struct AsymmetricCauchy {
    start_position: f64,
    beta: f64,
}

impl AsymmetricCauchy {
    /// Create a new asymmetric Cauchy process simulation
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position of the asymmetric Cauchy process.
    /// * `beta` - The asymmetry parameter of the asymmetric Cauchy process.
    pub fn new(start_position: impl Into<f64>, beta: impl Into<f64>) -> XResult<Self> {
        let start_position = start_position.into();
        let beta = beta.into();
        if !(-1.0..=1.0).contains(&beta) {
            return Err(SimulationError::InvalidParameters(format!(
                "The `beta` must be in the range [-1, 1], got {}",
                beta,
            ))
            .into());
        }
        Ok(Self {
            start_position,
            beta,
        })
    }

    /// Get the starting position of the asymmetric Cauchy process simulation
    pub fn start_position(&self) -> f64 {
        self.start_position
    }

    /// Get the asymmetry parameter of the asymmetric Cauchy process simulation
    pub fn asymmetry(&self) -> f64 {
        self.beta
    }
}

/// impl `ContinuousProcess` trait for AsymmetricCauchy process
impl ContinuousProcess for AsymmetricCauchy {
    /// Simulate asymmetric Cauchy process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the asymmetric Cauchy process simulation.
    /// * `time_step` - The time step of the asymmetric Cauchy process simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the asymmetric Cauchy process simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::AsymmetricCauchy;
    /// let cauchy = AsymmetricCauchy::new(0.0, 0.4).unwrap();
    /// let (t, x) = cauchy.simulate(10.0, 0.1).unwrap();
    /// ```
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        if time_step <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `time_step` must be positive, got {}",
                time_step
            ))
            .into());
        }
        let duration = duration.into();
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {}",
                duration
            ))
            .into());
        }
        simulate_asymmetric_cauchy(self.start_position, self.beta, duration, time_step)
    }
}

/// Simulate asymmetric Cauchy process
///
/// This function simulates asymmetric Cauchy process.
///
/// # Arguments
///
/// * `start_position` - The starting position of the asymmetric Cauchy process.
/// * `beta` - The asymmetry parameter of the asymmetric Cauchy process.
/// * `duration` - The duration of the asymmetric Cauchy process.
/// * `time_step` - The time step of the asymmetric Cauchy process.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::cauchy::simulate_asymmetric_cauchy;
/// let (t, x) = simulate_asymmetric_cauchy(0.0, 0.4, 10.0, 0.1).unwrap();
/// ```
pub fn simulate_asymmetric_cauchy(
    start_position: impl Into<f64>,
    beta: impl Into<f64>,
    duration: impl Into<f64>,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    simulate_asymmetric_levy(start_position, 1.0, beta, duration, time_step)
}

/// Cauchy process
///
/// This struct represents a Cauchy process.
///
/// # Fields
///
/// * `start_position` - The starting position of the Cauchy process.
#[derive(Debug, Clone)]
pub struct Cauchy {
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
    /// Create a new Cauchy process simulation
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position of the Cauchy process.
    pub fn new(start_position: impl Into<f64>) -> Self {
        let start_position = start_position.into();
        Self { start_position }
    }

    /// Get the starting position of the Cauchy process simulation
    pub fn start_position(&self) -> f64 {
        self.start_position
    }
}

/// impl `ContinuousProcess` trait for Cauchy process
impl ContinuousProcess for Cauchy {
    /// Simulate Cauchy process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the Cauchy process simulation.
    /// * `time_step` - The time step of the Cauchy process simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the Cauchy process simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::Cauchy;
    /// let cauchy = Cauchy::default();
    /// let (t, x) = cauchy.simulate(10.0, 0.1).unwrap();
    /// ```
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        if time_step <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `time_step` must be positive, got {}",
                time_step
            ))
            .into());
        }
        let duration = duration.into();
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {}",
                duration
            ))
            .into());
        }
        simulate_cauchy(self.start_position, duration, time_step)
    }
}

/// Simulate Cauchy process
///
/// This function simulates Cauchy process.
///
/// # Arguments
///
/// * `start_position` - The starting position of the Cauchy process.
/// * `duration` - The duration of the Cauchy process.
/// * `time_step` - The time step of the Cauchy process.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::cauchy::simulate_cauchy;
/// let (t, x) = simulate_cauchy(0.0, 1.0, 0.1).unwrap();
/// ```
pub fn simulate_cauchy(
    start_position: impl Into<f64>,
    duration: impl Into<f64>,
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
        println!("t: {:?}", t);
        println!("x: {:?}", x);
        let (t, x) = asymmetric_cauchy.simulate(duration, time_step).unwrap();
        println!("t: {:?}", t);
        println!("x: {:?}", x);
    }

    #[test]
    fn test_fpt() {
        let cauchy = Cauchy::default();
        let asymmetric_cauchy = AsymmetricCauchy::new(0.0, 0.4).unwrap();
        let time_step = 0.1;
        let fpt = cauchy.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {:?}", fpt);
        let fpt = asymmetric_cauchy
            .fpt((-1.0, 1.0), 1000.0, time_step)
            .unwrap();
        println!("fpt: {:?}", fpt);
    }

    #[test]
    fn test_occupation_time() {
        let cauchy = Cauchy::default();
        let asymmetric_cauchy = AsymmetricCauchy::new(0.0, 0.4).unwrap();
        let time_step = 0.1;
        let ot = cauchy
            .occupation_time((-1.0, 1.0), 10.0, time_step)
            .unwrap();
        println!("ot: {:?}", ot);
        let ot = asymmetric_cauchy
            .occupation_time((-1.0, 1.0), 10.0, time_step)
            .unwrap();
        println!("ot: {:?}", ot);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Cauchy>();
        assert_send_sync::<AsymmetricCauchy>();
    }
}
