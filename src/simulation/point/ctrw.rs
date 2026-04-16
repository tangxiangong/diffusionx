//! Continuous-time random walk simulation.

use crate::{
    SimulationError, XResult,
    random::{exponential, normal, stable},
    simulation::prelude::*,
    utils::cumsum,
};
use rand_distr::{Distribution, Exp1, StandardNormal, uniform::SampleUniform};

/// Continuous-time random walk.
///
/// # Mathematical Formulation
///
/// A continuous-time random walk (CTRW) generalizes a random walk by introducing
/// random waiting times between jumps. It can be written as
///
/// $$X(t) = X(0) + \sum_{i=1}^{N(t)} J_i,$$
///
/// where \(J_i\) are jump lengths and \(N(t)\) is the number of jumps completed
/// by time \(t\).
///
/// The waiting times between jumps typically follow a distribution with heavy tails, often
/// characterized by a power-law. When the waiting time distribution has infinite mean,
/// the resulting process exhibits subdiffusive behavior, with mean squared displacement
/// growing sublinearly with time:
///
/// $$\left\langle X^2(t)\right\rangle \sim t^\alpha,\qquad 0 < \alpha < 1.$$
///
/// CTRWs are widely used to model anomalous diffusion in complex systems, including
/// transport in disordered media, financial time series, and biological processes.
#[derive(Clone, Debug)]
pub struct CTRW<T: FloatExt = f64> {
    /// The alpha parameter of the stable distribution
    alpha: T,
    /// The beta parameter of the stable distribution
    beta: T,
    /// The starting position
    start_position: T,
}

impl<T: FloatExt> Default for CTRW<T> {
    fn default() -> Self {
        Self {
            alpha: T::one(),
            beta: T::from(2).unwrap(),
            start_position: T::zero(),
        }
    }
}

impl<T: FloatExt> CTRW<T> {
    /// Create a new `CTRW`
    ///
    /// # Arguments
    ///
    /// * `alpha` - The alpha parameter of the stable distribution.
    /// * `beta` - The beta parameter of the stable distribution.
    /// * `start_position` - The starting position of the process.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::point::CTRW;
    ///
    /// let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
    /// ```
    pub fn new(alpha: T, beta: T, start_position: T) -> XResult<Self> {
        if alpha <= T::zero() || alpha > T::one() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `alpha` must be between 0 and 1, got {alpha:?}"
            ))
            .into());
        }
        if beta <= T::zero() || beta > T::from(2).unwrap() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `beta` must be between 0 and 2, got {beta:?}"
            ))
            .into());
        }
        Ok(Self {
            alpha,
            beta,
            start_position,
        })
    }

    /// Get the stable index of the waiting time distribution
    pub fn get_alpha(&self) -> T {
        self.alpha
    }

    /// Get the stable index of the jump length distribution
    pub fn get_beta(&self) -> T {
        self.beta
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> T {
        self.start_position
    }
}

impl<T: FloatExt + SampleUniform> PointProcess<T> for CTRW<T>
where
    Exp1: Distribution<T>,
    StandardNormal: Distribution<T>,
{
    fn start(&self) -> T {
        self.start_position
    }

    /// Simulate the continuous-time random walk for a fixed number of jumps.
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::point::CTRW;
    ///
    /// let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
    /// let (t, x) = ctrw.simulate_with_step(1000).unwrap();
    /// ```
    fn simulate_with_step(&self, num_step: usize) -> XResult<(Vec<T>, Vec<T>)> {
        simulate_ctrw_with_step(self.alpha, self.beta, num_step, self.start_position)
    }
}

/// Simulate the continuous-time random walk for a fixed number of jumps.
///
/// # Arguments
///
/// * `alpha` - The alpha parameter of the stable distribution.
/// * `beta` - The beta parameter of the stable distribution.
/// * `num_step` - The number of steps.
/// * `start_position` - The starting position.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::point::ctrw::simulate_ctrw_with_step;
///
/// let (t, x) = simulate_ctrw_with_step(0.5, 1.0, 1000, 0.0).unwrap();
/// ```
pub fn simulate_ctrw_with_step<T: FloatExt + SampleUniform>(
    alpha: T,
    beta: T,
    num_step: usize,
    start_position: T,
) -> XResult<(Vec<T>, Vec<T>)>
where
    Exp1: Distribution<T>,
    StandardNormal: Distribution<T>,
{
    if alpha <= T::zero() || alpha > T::one() {
        return Err(SimulationError::InvalidParameters(format!(
            "The `alpha` must be between 0 and 1, got {alpha:?}"
        ))
        .into());
    }
    if beta <= T::zero() || beta > T::from(2).unwrap() {
        return Err(SimulationError::InvalidParameters(format!(
            "The `beta` must be between 0 and 2, got {beta:?}"
        ))
        .into());
    }
    if num_step == 0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `num_step` must be greater than 0, got {num_step}"
        ))
        .into());
    }
    let waiting_times = if alpha == T::one() {
        exponential::standard_rands(num_step)
    } else {
        stable::skew_rands(alpha, num_step)?
    };
    let jump_lengths = if beta == T::from(2).unwrap() {
        normal::standard_rands(num_step)
    } else {
        stable::sym_standard_rands(beta, num_step)?
    };
    let t = cumsum(T::zero(), &waiting_times);
    let x = cumsum(start_position, &jump_lengths);
    Ok((t, x))
}

/// Simulate the continuous-time random walk up to a fixed duration.
///
/// # Arguments
///
/// * `alpha` - The alpha parameter of the stable distribution.
/// * `beta` - The beta parameter of the stable distribution.
/// * `duration` - The duration of the simulation.
/// * `start_position` - The starting position.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::point::ctrw::simulate_ctrw_with_duration;
///
/// let (t, x) = simulate_ctrw_with_duration(0.5, 1.0, 10.0, 0.0).unwrap();
/// ```
pub fn simulate_ctrw_with_duration(
    alpha: f64,
    beta: f64,
    duration: f64,
    start_position: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    if duration <= 0.0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `duration` must be positive, got `{duration}`"
        ))
        .into());
    }
    let mut num_step = duration.ceil() as usize;
    let (t, x) = loop {
        let (t, x) = simulate_ctrw_with_step(alpha, beta, num_step, start_position)?;
        if t.last().is_none() {
            return Err(SimulationError::Unknown.into());
        }
        let end_time = *t.last().unwrap();
        if end_time >= duration {
            break (t, x);
        }
        num_step *= 2;
    };
    let index = t.iter().position(|&time| time >= duration).unwrap();
    let mut t_ = vec![0.0; index + 1];
    let mut x_ = vec![0.0; index + 1];
    t_[..index].copy_from_slice(&t[..index]);
    x_[..index].copy_from_slice(&x[..index]);
    if t[index] > duration {
        t_[index] = duration;
        x_[index] = x_[index - 1];
    } else {
        t_[index] = t[index];
        x_[index] = x[index];
    }

    Ok((t_, x_))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_ctrw_with_step() {
        let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
        let (t, x) = ctrw.simulate_with_step(1000).unwrap();
        assert_eq!(t.len(), 1001);
        assert_eq!(x.len(), 1001);
    }

    #[test]
    fn test_simulate_ctrw_with_duration() {
        let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
        let (_t, _x) = ctrw.simulate_with_duration(10.0).unwrap();
    }

    #[test]
    fn test_mean() {
        let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
        let _mean = ctrw.mean(1.0, 1000).unwrap();
    }

    #[test]
    fn test_msd() {
        let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
        let _msd = ctrw.msd(1.0, 1000).unwrap();
    }

    #[test]
    fn test_raw_moment() {
        let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
        let _moment = ctrw.raw_moment(1.0, 1, 1000).unwrap();
    }

    #[test]
    fn test_central_moment() {
        let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
        let _moment = ctrw.central_moment(1.0, 2, 1000).unwrap();
    }

    #[test]
    fn test_fpt() {
        let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
        let _fpt = ctrw.fpt((-1.0, 1.0), 1000.0).unwrap();
    }

    #[test]
    fn test_occupation_time() {
        let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
        let ot = ctrw.occupation_time((-1.0, 1.0), 1000.0).unwrap();
        assert!((0.0..=1000.0).contains(&ot));
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CTRW>();
    }
}
