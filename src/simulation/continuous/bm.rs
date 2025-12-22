//! Brownian motion simulation

use crate::{
    FloatExt, SimulationError, XResult, check_duration_time_step,
    random::{PAR_THRESHOLD, normal},
    simulation::prelude::*,
};
use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// Brownian motion
#[derive(Debug, Clone)]
pub struct Bm<T: FloatExt = f64> {
    /// The starting position
    start_position: T,
    /// The diffusion coefficient
    diffusion_coefficient: T,
}

impl<T: FloatExt> Default for Bm<T> {
    fn default() -> Self {
        Self {
            start_position: T::zero(),
            diffusion_coefficient: T::from(0.5).unwrap(),
        }
    }
}

impl<T: FloatExt> Bm<T> {
    /// Create a new `Bm`
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position.
    /// * `diffusion_coefficient` - The diffusion coefficient.
    pub fn new(start_position: T, diffusion_coefficient: T) -> XResult<Self> {
        if diffusion_coefficient <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The diffusion coefficient must be positive, got {diffusion_coefficient:?}"
            ))
            .into());
        }
        Ok(Self {
            start_position,
            diffusion_coefficient,
        })
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> T {
        self.start_position
    }

    /// Get the diffusion coefficient
    pub fn get_diffusion_coefficient(&self) -> T {
        self.diffusion_coefficient
    }
}

impl<T: FloatExt> ContinuousProcess<T> for Bm<T>
where
    StandardNormal: Distribution<T>,
{
    fn start(&self) -> T {
        self.start_position
    }

    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        simulate_bm(
            self.start_position,
            self.diffusion_coefficient,
            duration,
            time_step,
        )
    }

    fn displacement(&self, duration: T, time_step: T) -> XResult<T> {
        let two = T::from(2).unwrap();
        check_duration_time_step(duration, time_step)?;
        let num_steps = (duration / time_step).ceil().to_usize().unwrap();
        let std_dev = (two * self.diffusion_coefficient * time_step).sqrt();
        let normal = rand_distr::Normal::new(T::zero(), std_dev)?;
        let mut delta_x = if num_steps < PAR_THRESHOLD {
            let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
            (0..num_steps - 1).map(|_| rng.sample(normal)).sum()
        } else {
            (0..num_steps - 1)
                .into_par_iter()
                .map_init(
                    || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                    |r, _| r.sample(normal),
                )
                .sum()
        };
        let last_step = duration - T::from(num_steps - 1).unwrap() * time_step;
        delta_x +=
            (two * self.diffusion_coefficient * last_step).sqrt() * normal::standard_rand::<T>();
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
pub fn simulate_bm<T: FloatExt>(
    start_position: T,
    diffusion_coefficient: T,
    duration: T,
    time_step: T,
) -> XResult<(Vec<T>, Vec<T>)>
where
    StandardNormal: Distribution<T>,
{
    check_duration_time_step(duration, time_step)?;
    let two = T::from(2).unwrap();

    let num_steps = (duration / time_step).ceil().to_usize().unwrap();

    let std_dev = (two * diffusion_coefficient * time_step).sqrt();
    let noise = normal::rands(T::zero(), std_dev, num_steps - 1)?;

    let mut t = Vec::with_capacity(num_steps + 1);
    t.push(T::zero());

    let mut x = Vec::with_capacity(num_steps + 1);
    x.push(start_position);

    let mut current_x = start_position;
    let mut current_t = T::zero();
    for xi in noise {
        current_t += time_step;
        t.push(current_t);
        current_x += xi;
        x.push(current_x);
    }

    let last_step = duration - current_t;
    let sigma = (two * diffusion_coefficient * last_step).sqrt();
    let xi = normal::rand(T::zero(), sigma)?;
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
