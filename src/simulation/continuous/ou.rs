//! Ornstein-Uhlenbeck process simulation

use crate::{
    SimulationError, XResult,
    random::normal,
    simulation::prelude::*,
    utils::{diff, linspace},
};

/// Ornstein–Uhlenbeck process
///
/// dx(t) = -theta x(t) dt + sigma dW(t), x(0) = x0
///
/// where W(t) is the Wiener process, also called Brownian motion.
#[derive(Debug, Clone)]
pub struct OrnsteinUhlenbeck {
    /// The parameter controlling the strength of mean reversion.
    theta: f64,
    /// The diffusion coefficient controlling the noise intensity.
    sigma: f64,
    /// The starting position.
    start_position: f64,
}

impl OrnsteinUhlenbeck {
    /// Create a new `OrnsteinUhlenbeck`
    ///
    /// # Arguments
    ///
    /// * `theta` - The parameter controlling the strength of mean reversion.
    /// * `sigma` - The diffusion coefficient controlling the noise intensity.
    /// * `start_position` - The initial position x0 of the process.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::OrnsteinUhlenbeck;
    ///
    /// let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
    /// ```
    pub fn new(theta: f64, sigma: f64, start_position: impl Into<f64>) -> XResult<Self> {
        let start_position = start_position.into();
        Ok(Self {
            theta,
            sigma,
            start_position,
        })
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> f64 {
        self.start_position
    }

    /// Get the parameter controlling the strength of mean reversion
    pub fn get_theta(&self) -> f64 {
        self.theta
    }

    /// Get the diffusion coefficient
    pub fn get_sigma(&self) -> f64 {
        self.sigma
    }
}

impl ContinuousProcess for OrnsteinUhlenbeck {
    /// Simulate the Ornstein-Uhlenbeck process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration.
    /// * `time_step` - The time step.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::OrnsteinUhlenbeck, prelude::*};
    ///
    /// let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
    /// let (t, x) = ou.simulate(1.0, 0.01).unwrap();
    /// ```
    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_ou(
            self.theta,
            self.sigma,
            self.start_position,
            duration,
            time_step,
        )
    }
}

/// Simulate the Ornstein-Uhlenbeck process
///
/// # Mathematical Formulation
///
/// dx(t) = -theta x(t) dt + sigma dW(t), x(0) = x0
///
/// where W(t) is the Wiener process, also called Brownian motion.
///
/// # Arguments
///
/// * `theta` - The drift coefficient.
/// * `sigma` - The diffusion coefficient.
/// * `start_position` - The starting position.
/// * `duration` - The duration.
/// * `time_step` - The time step.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::ou::simulate_ou;
///
/// let (t, x) = simulate_ou(1.0, 1.0, 0.0, 1.0, 0.01).unwrap();
/// ```
pub fn simulate_ou(
    theta: f64,
    sigma: f64,
    start_position: f64,
    duration: f64,
    time_step: f64,
) -> XResult<Pair> {
    if duration <= 0.0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `duration` must be positive, got `{}`",
            duration
        ))
        .into());
    }
    if time_step <= 0.0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `time_step` must be positive, got `{}`",
            time_step
        ))
        .into());
    }
    if time_step > duration {
        return Err(SimulationError::InvalidParameters(format!(
            "The `time_step` must be less than or equal to the `duration`, got `{}` > `{}`",
            time_step, duration
        ))
        .into());
    }
    let t = linspace(0.0, duration, time_step);
    let num_steps = t.len() - 1;
    let noise = normal::standard_rands::<f64>(num_steps);
    let delta = diff(&t);

    let x = std::iter::once(start_position)
        .chain(
            noise
                .iter()
                .zip(delta)
                .scan(start_position, |state, (&xi, delta_t)| {
                    let current_x = *state;
                    let next_x =
                        current_x - theta * current_x * delta_t + sigma * xi * delta_t.sqrt();

                    *state = next_x;
                    Some(next_x)
                }),
        )
        .collect();
    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_ou() {
        let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
        let (t, x) = ou.simulate(1.0, 0.01).unwrap();
        assert_eq!(t.len(), x.len());
        assert!(t.last().unwrap() <= &1.0);
    }

    #[test]
    fn test_mean() {
        let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
        let _mean = ou.mean(1.0, 1000, 0.01).unwrap();
    }

    #[test]
    fn test_msd() {
        let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
        let msd = ou.msd(1.0, 1000, 0.01).unwrap();
        assert!(msd > 0.0);
    }

    #[test]
    fn test_raw_moment() {
        let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
        let _raw_moment = ou.raw_moment(1.0, 1, 1000, 0.01).unwrap();
        // 由于随机过程的特性，这里不做具体数值断言
    }

    #[test]
    fn test_central_moment() {
        let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
        let central_moment = ou.central_moment(1.0, 2, 1000, 0.01).unwrap();
        assert!(central_moment > 0.0);
    }

    #[test]
    fn test_fpt() {
        let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
        let _fpt = ou.fpt((-1.0, 1.0), 10.0, 0.01).unwrap();
    }

    #[test]
    fn test_occupation_time() {
        let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
        let occupation_time = ou.occupation_time((-1.0, 1.0), 1.0, 0.01).unwrap();
        assert!((0.0..=1.0).contains(&occupation_time));
    }
}
