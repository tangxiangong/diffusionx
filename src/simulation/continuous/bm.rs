//! Brownian motion simulation

use crate::{
    SimulationError, XResult,
    random::normal,
    simulation::prelude::*,
    utils::{cumsum, linspace},
};

/// Brownian motion
#[derive(Debug, Clone)]
pub struct Bm {
    /// The starting position
    start_position: f64,
    /// The diffusion coefficient
    diffusion_coefficient: f64,
}

impl Default for Bm {
    fn default() -> Self {
        Self {
            start_position: 0.0,
            diffusion_coefficient: 0.5,
        }
    }
}

impl Bm {
    /// Create a new `Bm`
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position.
    /// * `diffusion_coefficient` - The diffusion coefficient.
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

    /// Get the starting position
    pub fn get_start_position(&self) -> f64 {
        self.start_position
    }

    /// Get the diffusion coefficient
    pub fn get_diffusion_coefficient(&self) -> f64 {
        self.diffusion_coefficient
    }
}

impl ContinuousProcess for Bm {
    /// Simulate Brownian motion
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the trajectory.
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::Bm, prelude::*};
    ///
    /// let bm = Bm::default();
    /// let time_step = 0.1;
    /// let duration = 1.0;
    /// let (t, x) = bm.simulate(duration, time_step).unwrap();
    /// ```
    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_bm(
            self.start_position,
            self.diffusion_coefficient,
            duration,
            time_step,
        )
    }
}

/// Simulate Brownian motion
///
/// # Arguments
///
/// * `start_position` - The starting position of the Brownian motion.
/// * `diffusion_coefficient` - The diffusion coefficient of the Brownian motion.
/// * `duration` - The duration of the Brownian motion.
/// * `time_step` - The time step of the Brownian motion.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::bm::simulate_bm;
///
/// let start_position = 10.0;
/// let diffusion_coefficient = 1.0;
/// let time_step = 0.1;
/// let duration = 1.0;
/// let (t, x) = simulate_bm(start_position, diffusion_coefficient, duration, time_step).unwrap();
/// ```
pub fn simulate_bm(
    start_position: f64,
    diffusion_coefficient: f64,
    duration: f64,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let t = linspace(0.0, duration, time_step);
    let num_steps = t.len() - 1;
    let mut noise = normal::rands(0.0, 2.0 * diffusion_coefficient * time_step, num_steps)?;
    let last = match noise.last_mut() {
        Some(last) => last,
        None => return Err(SimulationError::Unknown.into()),
    };
    let delta = t[num_steps] - t[num_steps - 1];
    *last = normal::rand(0.0, 2.0 * diffusion_coefficient * delta)?;
    let x = cumsum(start_position, &noise);
    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::prelude::Moment;

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
