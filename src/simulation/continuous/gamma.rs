//! Gamma process simulation

use crate::{SimulationError, XResult, random::gamma, simulation::prelude::*};

/// Gamma process
///
/// # Mathematical Formulation
///
/// A Gamma process is a process that is non-negative and has a non-decreasing sample path with a Gamma distribution.
#[derive(Debug, Clone)]
pub struct Gamma {
    /// The shape parameter
    shape: f64,
    /// The rate parameter
    rate: f64,
}

impl Gamma {
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
    pub fn new(shape: impl Into<f64>, rate: impl Into<f64>) -> XResult<Self> {
        let shape = shape.into();
        let rate = rate.into();
        if shape <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `shape` must be positive, got {shape}"
            ))
            .into());
        }
        if rate <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `rate` must be positive, got {rate}"
            ))
            .into());
        }
        Ok(Self { shape, rate })
    }

    /// Get the shape parameter
    pub fn get_shape(&self) -> f64 {
        self.shape
    }

    /// Get the rate parameter
    pub fn get_rate(&self) -> f64 {
        self.rate
    }
}

impl ContinuousProcess for Gamma {
    fn simulate_unchecked(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_gamma(self.shape, self.rate, duration, time_step)
    }

    fn displacement(&self, duration: f64, time_step: f64) -> XResult<f64> {
        let num_steps = (duration / time_step).ceil() as usize;

        let scale = 1.0 / self.rate;
        let mut noise = gamma::rands(self.shape * time_step, scale, num_steps)?;
        let last_step = duration - ((num_steps - 1) as f64 * time_step);
        let noise_last = gamma::rand(self.shape * last_step, scale)?;
        *noise.last_mut().unwrap() = noise_last;

        Ok(noise.into_iter().sum())
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
pub fn simulate_gamma(
    shape: f64,
    rate: f64,
    duration: f64,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let num_steps = (duration / time_step).ceil() as usize;

    let scale = 1.0 / rate;
    let noise = gamma::rands(shape * time_step, scale, num_steps - 1)?;

    let mut t = Vec::with_capacity(num_steps + 1);
    let mut x = Vec::with_capacity(num_steps + 1);

    t.push(0.0);
    x.push(0.0);

    let mut current_x = 0.0;
    let mut current_t = 0.0;
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
