//! Geometric Brownian motion simulation
//!

use crate::{SimulationError, XResult, simulation::prelude::*};
use rayon::prelude::*;

use super::Bm;

/// Geometric Brownian motion
#[derive(Debug, Clone)]
pub struct GeometricBm {
    /// The starting position
    start_position: f64,
    /// The percentage drift
    mu: f64,
    /// The percentage volatility
    sigma: f64,
}

impl Default for GeometricBm {
    fn default() -> Self {
        Self {
            start_position: 0.0,
            mu: 0.0,
            sigma: 1.0,
        }
    }
}

impl GeometricBm {
    /// Create a new `GeometricBm`
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position.
    /// * `mu` - The percentage drift.
    /// * `sigma` - The percentage volatility.
    pub fn new(
        start_position: impl Into<f64>,
        mu: impl Into<f64>,
        sigma: impl Into<f64>,
    ) -> XResult<Self> {
        let start_position = start_position.into();
        let mu = mu.into();
        let sigma = sigma.into();
        if sigma <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The percentage volatility `sigma` must be positive, got {}",
                sigma
            ))
            .into());
        }
        Ok(Self {
            start_position,
            mu,
            sigma,
        })
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> f64 {
        self.start_position
    }

    /// Get the percentage drift
    pub fn get_mu(&self) -> f64 {
        self.mu
    }

    /// Get the percentage volatility
    pub fn get_sigma(&self) -> f64 {
        self.sigma
    }
}

impl ContinuousProcess for GeometricBm {
    /// Simulate geometric Brownian motion
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration.
    /// * `time_step` - The time step.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::GeometricBm, prelude::*};
    ///
    /// let gbm = GeometricBm::default();
    /// let time_step = 0.1;
    /// let duration = 1.0;
    /// let (t, x) = gbm.simulate(duration, time_step).unwrap();
    /// ```
    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_gbm(
            self.start_position,
            self.mu,
            self.sigma,
            duration,
            time_step,
        )
    }
}

/// Simulate geometric Brownian motion
///
/// # Arguments
///
/// * `start_position` - The starting position.
/// * `mu` - The percentage drift.
/// * `sigma` - The percentage volatility.
/// * `duration` - The duration.
/// * `time_step` - The time step.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::geometric_bm::simulate_gbm;
///
/// let start_position = 10.0;
/// let mu = 1.0;
/// let sigma = 1.0;
/// let time_step = 0.1;
/// let duration = 1.0;
/// let (t, x) = simulate_gbm(start_position, mu, sigma, duration, time_step).unwrap();
/// ```
pub fn simulate_gbm(
    start_position: f64,
    mu: f64,
    sigma: f64,
    duration: f64,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let bm = Bm::default();
    let (t, b) = bm.simulate(duration, time_step)?;
    let tmp = mu - sigma * sigma / 2.0;
    let x = t
        .par_iter()
        .zip(b)
        .map(|(t_i, b_i)| start_position * (tmp * t_i + sigma * b_i).exp())
        .collect();
    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::prelude::Moment;

    #[test]
    fn test_simulate_gbm() {
        let gbm = GeometricBm::new(10.0, 1.0, 1.0).unwrap();
        let duration = 1.0;
        let time_step = 0.1;
        let (t, x) = gbm.simulate(duration, time_step).unwrap();
        println!("t: {:?}", t);
        println!("x: {:?}", x);
    }

    #[test]
    fn test_raw_moment() {
        let gbm = GeometricBm::new(10.0, 1.0, 1.0).unwrap();
        let duration = 1.0;
        let time_step = 0.1;
        let traj = gbm.duration(duration).unwrap();
        let moment = traj.raw_moment(1, 1000, time_step).unwrap();
        println!("moment: {:?}", moment);
    }

    #[test]
    fn test_fpt() {
        let gbm = GeometricBm::new(10.0, 1.0, 1.0).unwrap();
        let time_step = 0.1;
        let fpt = gbm.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {:?}", fpt);
    }

    #[test]
    fn test_occupation_time() {
        let gbm = GeometricBm::new(10.0, 1.0, 1.0).unwrap();
        let time_step = 0.1;
        let ot = gbm.occupation_time((-1.0, 1.0), 10.0, time_step).unwrap();
        println!("ot: {:?}", ot);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GeometricBm>();
    }
}
