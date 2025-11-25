//! Brownian motion simulation

use crate::{
    SimulationError, XResult, check_duration_time_step, random::normal, simulation::prelude::*,
};
use rand::{Rng, rng};
use rayon::prelude::*;

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
                "The diffusion coefficient must be positive, got {diffusion_coefficient}"
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
    fn start(&self) -> f64 {
        self.start_position
    }

    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_bm(
            self.start_position,
            self.diffusion_coefficient,
            duration,
            time_step,
        )
    }

    fn displacement(&self, duration: f64, time_step: f64) -> XResult<f64> {
        let num_steps = (duration / time_step).ceil() as usize;
        let std_dev = (2.0 * self.diffusion_coefficient * time_step).sqrt();
        let normal = rand_distr::Normal::new(0.0, std_dev)?;
        let mut delta_x = (0..num_steps - 1)
            .into_par_iter()
            .map_init(rng, |r, _| r.sample(normal))
            .sum();
        let last_step = duration - (num_steps - 1) as f64 * time_step;
        delta_x +=
            (2.0 * self.diffusion_coefficient * last_step).sqrt() * normal::standard_rand::<f64>();
        Ok(delta_x)
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
    check_duration_time_step(duration, time_step)?;

    let num_steps = (duration / time_step).ceil() as usize;

    let std_dev = (2.0 * diffusion_coefficient * time_step).sqrt();
    let noise = normal::rands(0.0, std_dev, num_steps - 1)?;

    let mut t = Vec::with_capacity(num_steps + 1);
    t.push(0.0);

    let mut x = Vec::with_capacity(num_steps + 1);
    x.push(start_position);

    let mut current_x = start_position;
    let mut current_t = 0.0;
    for xi in noise {
        current_t += time_step;
        t.push(current_t);
        current_x += xi;
        x.push(current_x);
    }

    let last_step = duration - current_t;
    let sigma = (2.0 * diffusion_coefficient * last_step).sqrt();
    let xi = normal::rand::<f64>(0.0, sigma)?;
    current_x += xi;
    x.push(current_x);
    t.push(duration);

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
        println!("t: {t:?}");
        println!("x: {x:?}");
    }

    #[test]
    fn test_msd() {
        let bm = Bm::default();
        let m = bm.msd(100.0, 10_000, 0.01).unwrap();
        println!("{m}");
    }

    #[test]
    fn test_raw_moment() {
        let bm = Bm::new(10.0, 1.0).unwrap();
        let duration = 1.0;
        let time_step = 0.1;
        let traj = bm.duration(duration).unwrap();
        let moment = traj.raw_moment(1, 1000, time_step).unwrap();
        println!("moment: {moment:?}");
    }

    #[test]
    fn test_fpt() {
        let bm = Bm::new(0.0, 1.0).unwrap();
        let time_step = 0.1;
        let fpt = bm.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {fpt:?}");
    }

    #[test]
    fn test_occupation_time() {
        let bm = Bm::new(0.0, 1.0).unwrap();
        let time_step = 0.1;
        let ot = bm.occupation_time((-1.0, 1.0), 10.0, time_step).unwrap();
        println!("ot: {ot:?}");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Bm>();
    }
}
