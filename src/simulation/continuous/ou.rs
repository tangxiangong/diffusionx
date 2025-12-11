//! Ornstein-Uhlenbeck process simulation

use crate::{FloatExt, XResult, check_duration_time_step, random::normal, simulation::prelude::*};
use rand_distr::{Distribution, StandardNormal};

/// Ornstein–Uhlenbeck process
///
/// $$dx(t) = -\theta x(t) dt + \sigma dW(t),\qquad x(0) = x_0$$
///
/// where $W(t)$ is the Wiener process, also called Brownian motion.
#[derive(Debug, Clone)]
pub struct OrnsteinUhlenbeck<T: FloatExt = f64> {
    /// The parameter controlling the strength of mean reversion.
    theta: T,
    /// The diffusion coefficient controlling the noise intensity.
    sigma: T,
    /// The starting position.
    start_position: T,
}

impl<T: FloatExt> Default for OrnsteinUhlenbeck<T> {
    fn default() -> Self {
        Self {
            theta: T::one(),
            sigma: T::one(),
            start_position: T::zero(),
        }
    }
}

impl<T: FloatExt> OrnsteinUhlenbeck<T> {
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
    /// use diffusionx::simulation::continuous::ou::OrnsteinUhlenbeck;
    ///
    /// let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
    /// ```
    pub fn new(theta: T, sigma: T, start_position: T) -> XResult<Self> {
        Ok(Self {
            theta,
            sigma,
            start_position,
        })
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> T {
        self.start_position
    }

    /// Get the parameter controlling the strength of mean reversion
    pub fn get_theta(&self) -> T {
        self.theta
    }

    /// Get the diffusion coefficient
    pub fn get_sigma(&self) -> T {
        self.sigma
    }
}

impl<T: FloatExt> ContinuousProcess<T> for OrnsteinUhlenbeck<T>
where
    StandardNormal: Distribution<T>,
{
    fn start(&self) -> T {
        self.start_position
    }

    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        simulate_ou(
            self.theta,
            self.sigma,
            self.start_position,
            duration,
            time_step,
        )
    }

    fn displacement(&self, duration: T, time_step: T) -> XResult<T> {
        check_duration_time_step(duration, time_step)?;

        let num_steps = (duration / time_step).ceil().to_usize().unwrap();

        let mut scale = time_step.sqrt();

        let mut current_x = self.start_position;
        let mut mu;

        let noises = normal::standard_rands::<T>(num_steps - 1);

        for xi in noises {
            mu = -self.theta * current_x;
            current_x += mu * time_step + self.sigma * xi * scale;
        }

        let last_step = duration - T::from(num_steps - 1).unwrap() * time_step;
        scale = last_step.sqrt();
        mu = -self.theta * current_x;
        current_x += mu * last_step + self.sigma * normal::standard_rand::<T>() * scale;

        Ok(current_x - self.start_position)
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
pub fn simulate_ou<T: FloatExt>(
    theta: T,
    sigma: T,
    start_position: T,
    duration: T,
    time_step: T,
) -> XResult<(Vec<T>, Vec<T>)>
where
    StandardNormal: Distribution<T>,
{
    check_duration_time_step(duration, time_step)?;

    let num_steps = (duration / time_step).ceil().to_usize().unwrap();

    let mut t = Vec::with_capacity(num_steps + 1);
    let mut x = Vec::with_capacity(num_steps + 1);

    t.push(T::zero());
    x.push(start_position);

    let mut scale = time_step.sqrt();

    let mut current_t = T::zero();
    let mut current_x = start_position;
    let mut mu;

    let noises = normal::standard_rands::<T>(num_steps - 1);

    for xi in noises {
        mu = -theta * current_x;
        current_x += mu * time_step + sigma * xi * scale;
        current_t += time_step;
        t.push(current_t);
        x.push(current_x);
    }

    let last_step = duration - current_t;
    scale = last_step.sqrt();
    mu = -theta * current_x;
    current_x += mu * last_step + sigma * normal::standard_rand::<T>() * scale;
    x.push(current_x);
    t.push(duration);

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
