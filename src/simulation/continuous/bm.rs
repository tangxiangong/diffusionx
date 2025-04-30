//! Brownian motion simulation
//!
//! For Levy process, see [`crate::simulation::continuous::levy`].

use crate::{SimulationError, XResult, random::normal, simulation::prelude::*, utils::cumsum};
use rayon::prelude::*;

/// Brownian motion
///
/// This struct represents a Brownian motion.
///
/// # Fields
///
/// * `start_position` - The starting position of the Brownian motion.
/// * `diffusion_coefficient` - The diffusion coefficient of the Brownian motion.
#[derive(Debug, Clone)]
pub struct Bm {
    start_position: f64,
    diffusion_coefficient: f64,
}

impl Default for Bm {
    fn default() -> Self {
        Self {
            start_position: 0.0,
            diffusion_coefficient: 1.0,
        }
    }
}

impl Bm {
    /// Create a new Brownian motion simulation
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position of the Brownian motion.
    /// * `diffusion_coefficient` - The diffusion coefficient of the Brownian motion.
    pub fn new(
        start_position: impl Into<f64>,
        diffusion_coefficient: impl Into<f64>,
    ) -> XResult<Self> {
        let start_position = start_position.into();
        let diffusion_coefficient = diffusion_coefficient.into();
        if diffusion_coefficient <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The diffusion coefficient must be positive, got {}",
                diffusion_coefficient
            ))
            .into());
        }
        Ok(Self {
            start_position,
            diffusion_coefficient,
        })
    }

    /// Get the starting position of the Brownian motion simulation
    pub fn start_position(&self) -> f64 {
        self.start_position
    }

    /// Get the diffusion coefficient of the Brownian motion simulation
    pub fn diffusion_coefficient(&self) -> f64 {
        self.diffusion_coefficient
    }
}

/// impl `ContinuousProcess` trait for Brownian motion
impl ContinuousProcess for Bm {
    /// Simulate Brownian motion
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the Brownian motion simulation.
    /// * `time_step` - The time step of the Brownian motion simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the Brownian motion simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::Bm, prelude::*};
    /// let bm = Bm::default();
    /// let time_step = 0.1;
    /// let duration = 1.0;
    /// let (t, x) = bm.simulate(duration, time_step).unwrap();
    /// ```
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        simulate_bm(
            self.start_position,
            self.diffusion_coefficient,
            duration.into(),
            time_step,
        )
    }
}

/// Simulate Brownian motion
///
/// This function simulates Brownian motion.
///
/// # Arguments
///
/// * `start_position` - The starting position of the Brownian motion.
/// * `diffusion_coefficient` - The diffusion coefficient of the Brownian motion.
/// * `duration` - The duration of the Brownian motion.
/// * `time_step` - The time step of the Brownian motion.
///
/// # Returns
///
/// A tuple containing the time and the position of the Brownian motion simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::bm::simulate_bm;
/// let start_position = 10.0;
/// let diffusion_coefficient = 1.0;
/// let time_step = 0.1;
/// let duration = 1.0;
/// let (t, x) = simulate_bm(start_position, diffusion_coefficient, duration, time_step).unwrap();
/// ```
pub fn simulate_bm(
    start_position: impl Into<f64>,
    diffusion_coefficient: impl Into<f64>,
    duration: impl Into<f64>,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let start_position = start_position.into();
    let diffusion_coefficient = diffusion_coefficient.into();
    let duration = duration.into();
    let num_steps = (duration / time_step).ceil() as usize;
    let t = (0..=num_steps)
        .into_par_iter()
        .map(|i| time_step * i as f64)
        .collect::<Vec<_>>();
    let noise = normal::rands(0.0, 2.0 * diffusion_coefficient * time_step, num_steps)?;
    let x = cumsum(start_position, &noise);
    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::prelude::{ContinuousTrajectoryTrait, Moment};

    #[test]
    fn test_simulate_bm() {
        let bm = Bm::new(10.0, 1.0).unwrap();
        let duration = 1.0;
        let time_step = 0.1;
        let (t, x) = bm.simulate(duration, time_step).unwrap();
        println!("t: {:?}", t);
        println!("x: {:?}", x);
    }

    #[test]
    fn test_raw_moment() {
        let bm = Bm::new(10.0, 1.0).unwrap();
        let duration = 1.0;
        let time_step = 0.1;
        let traj = bm.duration(duration).unwrap();
        let moment = traj.raw_moment(1, 1000, time_step).unwrap();
        println!("moment: {:?}", moment);
    }

    #[test]
    fn test_fpt() {
        let bm = Bm::new(0.0, 1.0).unwrap();
        let time_step = 0.1;
        let fpt = bm.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {:?}", fpt);
    }

    #[test]
    fn test_occupation_time() {
        let bm = Bm::new(0.0, 1.0).unwrap();
        let time_step = 0.1;
        let ot = bm.occupation_time((-1.0, 1.0), 10.0, time_step).unwrap();
        println!("ot: {:?}", ot);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Bm>();
    }
}
