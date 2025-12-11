//! Brownian bridge simulation

use crate::{
    FloatExt, SimulationError, XResult, check_duration_time_step,
    simulation::{continuous::Bm, prelude::*},
};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Brownian bridge
#[derive(Debug, Clone)]
pub struct BrownianBridge<T: FloatExt = f64> {
    _marker: std::marker::PhantomData<T>,
}

impl<T: FloatExt> Default for BrownianBridge<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatExt> BrownianBridge<T> {
    /// Create a new `BrownianBridge`
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::brownian_bridge::BrownianBridge;
    ///
    /// let bb = BrownianBridge::new();
    /// ```
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: FloatExt> ContinuousProcess<T> for BrownianBridge<T>
where
    StandardNormal: Distribution<T>,
{
    fn start(&self) -> T {
        T::zero()
    }

    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        simulate_brownian_bridge(duration, time_step)
    }

    fn displacement(&self, _: T, _: T) -> XResult<T> {
        Ok(T::zero())
    }
}

/// Simulate Brownian bridge
///
/// # Arguments
///
/// * `duration` - The duration of the trajectory.
/// * `time_step` - The time step of the simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::brownian_bridge::simulate_brownian_bridge;
///
/// let time_step = 0.1;
/// let duration = 1.0;
/// let (t, x) = simulate_brownian_bridge(duration, time_step).unwrap();
/// ```
pub fn simulate_brownian_bridge<T: FloatExt>(duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)>
where
    StandardNormal: Distribution<T>,
{
    check_duration_time_step(duration, time_step)?;

    let bm = Bm::default();
    let (t, traj) = bm.simulate(duration, time_step)?;
    let end_position = match traj.last() {
        Some(x) => *x,
        None => return Err(SimulationError::Unknown.into()),
    };
    let x = traj
        .into_par_iter()
        .zip(t.par_iter())
        .map(|(traj_i, &t_i)| traj_i - end_position * t_i / duration)
        .collect();
    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::prelude::Moment;

    #[test]
    fn test_simulate_bb() {
        let bb = BrownianBridge::new();
        let duration = 1.0;
        let time_step = 0.1;
        let (t, x) = bb.simulate(duration, time_step).unwrap();
        println!("t: {t:?}");
        println!("x: {x:?}");
    }

    #[test]
    fn test_raw_moment() {
        let bb = BrownianBridge::new();
        let duration = 1.0;
        let time_step = 0.1;
        let traj = bb.duration(duration).unwrap();
        let moment = traj.raw_moment(1, 1000, time_step).unwrap();
        println!("moment: {moment:?}");
    }

    #[test]
    fn test_fpt() {
        let bb = BrownianBridge::new();
        let time_step = 0.1;
        let fpt = bb.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {fpt:?}");
    }

    #[test]
    fn test_occupation_time() {
        let bb = BrownianBridge::new();
        let time_step = 0.1;
        let ot = bb.occupation_time((-1.0, 1.0), 10.0, time_step).unwrap();
        println!("ot: {ot:?}");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BrownianBridge>();
    }
}
