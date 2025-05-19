//! Lévy process
//!
//! The Lévy process is a process with independent and stationary increments.
//!

use crate::{SimulationError, XResult, random::stable, simulation::prelude::*, utils::cumsum};
use rayon::prelude::*;

/// Asymmetric Lévy process
#[derive(Debug, Clone)]
pub struct AsymmetricLevy {
    /// The starting position
    start_position: f64,
    /// The stability index
    alpha: f64,
    /// The asymmetry parameter
    beta: f64,
}

impl AsymmetricLevy {
    /// Create a new `AsymmetricLevy`
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position.
    /// * `alpha` - The stability index.
    /// * `beta` - The asymmetry parameter.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::AsymmetricLevy;
    ///
    /// let levy = AsymmetricLevy::new(0.0, 1.5, 0.4).unwrap();
    /// ```
    pub fn new(
        start_position: impl Into<f64>,
        alpha: impl Into<f64>,
        beta: impl Into<f64>,
    ) -> XResult<Self> {
        let start_position = start_position.into();
        let alpha = alpha.into();
        if alpha <= 0.0 || alpha > 2.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `alpha` must be in the range (0, 2], got {}",
                alpha
            ))
            .into());
        }
        let beta = beta.into();
        if !(-1.0..=1.0).contains(&beta) {
            return Err(SimulationError::InvalidParameters(format!(
                "The `beta` must be in the range [-1, 1], got {}",
                beta
            ))
            .into());
        }
        Ok(Self {
            start_position,
            alpha,
            beta,
        })
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> f64 {
        self.start_position
    }

    /// Get the stability index
    pub fn get_alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the asymmetry parameter
    pub fn get_beta(&self) -> f64 {
        self.beta
    }
}

/// impl `ContinuousProcess` trait for `AsymmetricLevy`
impl ContinuousProcess for AsymmetricLevy {
    /// Simulate the asymmetric Lévy process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration.
    /// * `time_step` - The time step.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::AsymmetricLevy, prelude::*};
    ///
    /// let levy = AsymmetricLevy::new(0.0, 1.5, 0.4).unwrap();
    /// let (t, x) = levy.simulate(10.0, 0.1).unwrap();
    /// ```
    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        if time_step <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `time_step` must be positive, got {}",
                time_step
            ))
            .into());
        }
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {}",
                duration
            ))
            .into());
        }
        simulate_asymmetric_levy(
            self.start_position,
            self.alpha,
            self.beta,
            duration,
            time_step,
        )
    }
}

/// Simulate the asymmetric Lévy process
///
/// # Arguments
///
/// * `start_position` - The starting position.
/// * `alpha` - The stability index.
/// * `beta` - The asymmetry parameter.
/// * `duration` - The duration.
/// * `time_step` - The time step.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::{continuous::AsymmetricLevy, prelude::*};
///
/// let levy = AsymmetricLevy::new(0.0, 1.5, 0.4).unwrap();
/// let (t, x) = levy.simulate(10.0, 0.1).unwrap();
/// ```
pub fn simulate_asymmetric_levy(
    start_position: f64,
    alpha: f64,
    beta: f64,
    duration: f64,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let num_steps = (duration / time_step).ceil() as usize;
    let t = (0..=num_steps)
        .into_par_iter()
        .map(|i| time_step * i as f64)
        .collect::<Vec<_>>();
    let noise = stable::standard_rands(alpha, beta, num_steps as usize)?
        .into_par_iter()
        .map(|x| x * time_step.powf(1.0 / alpha))
        .collect::<Vec<_>>();
    let x = cumsum(start_position, &noise);
    Ok((t, x))
}

/// Lévy process
#[derive(Debug, Clone)]
pub struct Levy {
    /// The starting position
    start_position: f64,
    /// The stability index
    alpha: f64,
}

impl Levy {
    /// Create a new `Levy`
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position.
    /// * `alpha` - The stability index.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::Levy;
    ///
    /// let levy = Levy::new(0.0, 1.5).unwrap();
    /// ```
    pub fn new(start_position: impl Into<f64>, alpha: impl Into<f64>) -> XResult<Self> {
        let start_position = start_position.into();
        let alpha = alpha.into();
        if alpha <= 0.0 || alpha > 2.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `alpha` must be in the range (0, 2], got {}",
                alpha
            ))
            .into());
        }
        Ok(Self {
            start_position,
            alpha,
        })
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> f64 {
        self.start_position
    }

    /// Get the stability index
    pub fn get_alpha(&self) -> f64 {
        self.alpha
    }
}

/// impl `ContinuousProcess` trait for `Levy`
impl ContinuousProcess for Levy {
    /// Simulate the Lévy process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration.
    /// * `time_step` - The time step.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::Levy, prelude::*};
    ///
    /// let levy = Levy::new(0.0, 1.5).unwrap();
    /// let (t, x) = levy.simulate(1.0, 0.1).unwrap();
    /// ```
    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        if time_step <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `time_step` must be positive, got {}",
                time_step
            ))
            .into());
        }
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {}",
                duration
            ))
            .into());
        }
        simulate_levy(self.start_position, self.alpha, duration, time_step)
    }
}

/// Simulate the Lévy process
///
/// # Arguments
///
/// * `start_position` - The starting position.
/// * `alpha` - The stability index.
/// * `duration` - The duration.
/// * `time_step` - The time step.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::levy::simulate_levy;
///
/// let (t, x) = simulate_levy(0.0, 1.5, 1.0, 0.1).unwrap();
/// ```
pub fn simulate_levy(
    start_position: f64,
    alpha: f64,
    duration: f64,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let num_steps = (duration / time_step).ceil() as usize;
    let t = (0..=num_steps)
        .into_par_iter()
        .map(|i| time_step * i as f64)
        .collect::<Vec<_>>();
    let noise = stable::standard_rands(alpha, 0.0, num_steps as usize)?
        .into_par_iter()
        .map(|x| x * time_step.powf(1.0 / alpha))
        .collect::<Vec<_>>();
    let x = cumsum(start_position, &noise);
    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_levy() {
        let levy = Levy::new(10.0, 1.5).unwrap();
        let asymmetric_levy = AsymmetricLevy::new(10.0, 1.5, 0.4).unwrap();
        let time_step = 0.1;
        let duration = 1.0;
        let (t, x) = levy.simulate(duration, time_step).unwrap();
        println!("t: {:?}", t);
        println!("x: {:?}", x);
        let (t, x) = asymmetric_levy.simulate(duration, time_step).unwrap();
        println!("t: {:?}", t);
        println!("x: {:?}", x);
    }

    #[test]
    fn test_fpt() {
        let levy = Levy::new(0.0, 1.5).unwrap();
        let asymmetric_levy = AsymmetricLevy::new(0.0, 1.5, 0.4).unwrap();
        let time_step = 0.1;
        let fpt = levy.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {:?}", fpt);
        let fpt = asymmetric_levy.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {:?}", fpt);
    }

    #[test]
    fn test_occupation_time() {
        let levy = Levy::new(0.0, 1.5).unwrap();
        let asymmetric_levy = AsymmetricLevy::new(0.0, 1.5, 0.4).unwrap();
        let time_step = 0.1;
        let ot = levy.occupation_time((-1.0, 1.0), 10.0, time_step).unwrap();
        println!("ot: {:?}", ot);
        let ot = asymmetric_levy
            .occupation_time((-1.0, 1.0), 10.0, time_step)
            .unwrap();
        println!("ot: {:?}", ot);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Levy>();
        assert_send_sync::<AsymmetricLevy>();
    }
}
