//! Geometric Brownian motion simulation

use crate::{
    FloatExt, SimulationError, XResult, check_duration_time_step,
    random::PAR_THRESHOLD,
    simulation::{continuous::Bm, prelude::*},
};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Geometric Brownian motion.
///
/// The process solves
///
/// $$dX(t) = \mu X(t)\,dt + \sigma X(t)\,dW(t),$$
///
/// with explicit solution
///
/// $$X(t) = X(0)\exp\left((\mu-\tfrac{1}{2}\sigma^2)t+\sigma W(t)\right).$$
#[derive(Debug, Clone)]
pub struct GeometricBm<T: FloatExt = f64> {
    /// The starting position
    start_position: T,
    /// The percentage drift
    mu: T,
    /// The percentage volatility
    sigma: T,
}

impl<T: FloatExt> Default for GeometricBm<T> {
    fn default() -> Self {
        Self {
            start_position: T::zero(),
            mu: T::zero(),
            sigma: T::one(),
        }
    }
}

impl<T: FloatExt> GeometricBm<T> {
    /// Create a new `GeometricBm`
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position.
    /// * `mu` - The percentage drift.
    /// * `sigma` - The percentage volatility.
    pub fn new(start_position: T, mu: T, sigma: T) -> XResult<Self> {
        if sigma <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The percentage volatility `sigma` must be positive, got {sigma:?}"
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
    pub fn get_start_position(&self) -> T {
        self.start_position
    }

    /// Get the percentage drift
    pub fn get_mu(&self) -> T {
        self.mu
    }

    /// Get the percentage volatility
    pub fn get_sigma(&self) -> T {
        self.sigma
    }
}

impl<T: FloatExt> ContinuousProcess<T> for GeometricBm<T>
where
    StandardNormal: Distribution<T>,
{
    fn start(&self) -> T {
        self.start_position
    }

    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        simulate_gbm(
            self.start_position,
            self.mu,
            self.sigma,
            duration,
            time_step,
        )
    }

    fn displacement(&self, duration: T, time_step: T) -> XResult<T> {
        let bm = Bm::default();
        let b = bm.displacement(duration, time_step)?;
        let tmp = self.mu - self.sigma * self.sigma / T::from(2).unwrap();
        let end_position = self.start_position * (tmp * duration + self.sigma * b).exp();
        Ok(end_position - self.start_position)
    }
}

/// Simulate geometric Brownian motion.
///
/// The generated path uses the exact representation
///
/// $$X(t) = X(0)\exp\left((\mu-\tfrac{1}{2}\sigma^2)t+\sigma W(t)\right).$$
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
pub fn simulate_gbm<T: FloatExt>(
    start_position: T,
    mu: T,
    sigma: T,
    duration: T,
    time_step: T,
) -> XResult<(Vec<T>, Vec<T>)>
where
    StandardNormal: Distribution<T>,
{
    check_duration_time_step(duration, time_step)?;

    let bm = Bm::default();
    let (t, b) = bm.simulate(duration, time_step)?;
    let tmp = mu - sigma * sigma / T::from(2).unwrap();
    let x = if t.len() < PAR_THRESHOLD {
        t.iter()
            .zip(b)
            .map(|(&t_i, b_i)| start_position * (tmp * t_i + sigma * b_i).exp())
            .collect()
    } else {
        t.par_iter()
            .zip(b)
            .map(|(&t_i, b_i)| start_position * (tmp * t_i + sigma * b_i).exp())
            .collect()
    };
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
        println!("t: {t:?}");
        println!("x: {x:?}");
    }

    #[test]
    fn test_raw_moment() {
        let gbm = GeometricBm::new(10.0, 1.0, 1.0).unwrap();
        let duration = 1.0;
        let time_step = 0.1;
        let traj = gbm.duration(duration).unwrap();
        let moment = traj.raw_moment(1, 1000, time_step).unwrap();
        println!("moment: {moment:?}");
    }

    #[test]
    fn test_displacement_is_terminal_change() {
        let gbm = GeometricBm::new(10.0f64, 0.0, 1e-12).unwrap();
        let displacement = gbm.displacement(1.0, 0.1).unwrap();
        assert!(
            displacement.abs() < 1e-6,
            "displacement should be terminal change, got {displacement}"
        );
    }

    #[test]
    fn test_fpt() {
        let gbm = GeometricBm::new(10.0, 1.0, 1.0).unwrap();
        let time_step = 0.1;
        let fpt = gbm.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {fpt:?}");
    }

    #[test]
    fn test_occupation_time() {
        let gbm = GeometricBm::new(10.0, 1.0, 1.0).unwrap();
        let time_step = 0.1;
        let ot = gbm.occupation_time((-1.0, 1.0), 10.0, time_step).unwrap();
        println!("ot: {ot:?}");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GeometricBm>();
    }
}
