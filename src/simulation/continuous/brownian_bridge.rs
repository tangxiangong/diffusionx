//! Brownian bridge simulation
//!

use crate::{SimulationError, XResult, simulation::prelude::*};
use rayon::prelude::*;

use super::Bm;

/// Brownian bridge
#[derive(Debug, Clone)]
pub struct BrownianBridge;

/// impl `ContinuousProcess` trait for `BrownianBridge`
impl ContinuousProcess for BrownianBridge {
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
    /// use diffusionx::simulation::{continuous::BrownianBridge, prelude::*};
    ///
    /// let bb = BrownianBridge;
    /// let time_step = 0.1;
    /// let duration = 1.0;
    /// let (t, x) = bb.simulate(duration, time_step).unwrap();
    /// ```
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        simulate_brownian_bridge(duration.into(), time_step)
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
pub fn simulate_brownian_bridge(
    duration: impl Into<f64>,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let duration: f64 = duration.into();
    let bm = Bm::default();
    let (t, traj) = bm.simulate(duration, time_step)?;
    let end_position = match traj.last() {
        Some(x) => *x,
        None => return Err(SimulationError::Unknown.into()),
    };
    let x = traj
        .into_par_iter()
        .zip(t.par_iter())
        .map(|(traj_i, t_i)| traj_i - end_position * t_i / duration)
        .collect();
    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::prelude::Moment;

    #[test]
    fn test_simulate_bb() {
        let bb = BrownianBridge;
        let duration = 1.0;
        let time_step = 0.1;
        let (t, x) = bb.simulate(duration, time_step).unwrap();
        println!("t: {:?}", t);
        println!("x: {:?}", x);
    }

    #[test]
    fn test_raw_moment() {
        let bb = BrownianBridge;
        let duration = 1.0;
        let time_step = 0.1;
        let traj = bb.duration(duration).unwrap();
        let moment = traj.raw_moment(1, 1000, time_step).unwrap();
        println!("moment: {:?}", moment);
    }

    #[test]
    fn test_fpt() {
        let bb = BrownianBridge;
        let time_step = 0.1;
        let fpt = bb.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {:?}", fpt);
    }

    #[test]
    fn test_occupation_time() {
        let bb = BrownianBridge;
        let time_step = 0.1;
        let ot = bb.occupation_time((-1.0, 1.0), 10.0, time_step).unwrap();
        println!("ot: {:?}", ot);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BrownianBridge>();
    }
}
