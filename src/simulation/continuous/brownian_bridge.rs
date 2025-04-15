//! Brownian bridge simulation
//!
//! For Brownian motion, see [`crate::simulation::bm`].

use crate::{SimulationError, XResult, simulation::prelude::*};
use rayon::prelude::*;

use super::Bm;

/// Brownian bridge
///
/// This struct represents a Brownian bridge.
#[derive(Debug, Clone)]
pub struct BrownianBridge;

impl BrownianBridge {
    /// Get the mean of the Brownian bridge simulation
    ///
    /// # Returns
    ///
    /// A f64 representing the mean of the Brownian bridge simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::BrownianBridge;
    /// let bb = BrownianBridge;
    /// let mean = bb.mean(1.0, 1000, 0.1).unwrap();
    /// ```
    pub fn mean(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let traj = self.duration(duration).unwrap();
        traj.raw_moment(1, particles, time_step)
    }

    /// Get the mean square displacement of the Brownian bridge simulation
    ///
    /// # Returns
    ///
    /// A f64 representing the mean square displacement of the Brownian bridge simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let bb = BrownianBridge;
    /// let msd = bb.msd(1.0, 1000, 0.1).unwrap();
    /// ```
    pub fn msd(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let traj = self.duration(duration).unwrap();
        traj.central_moment(2, particles, time_step)
    }

    /// Get the raw moment of the Brownian bridge simulation
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the Brownian bridge simulation.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the Brownian bridge simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the raw moment of the Brownian bridge simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let bb = BrownianBridge;
    /// let moment = bb.raw_moment(1.0, 1000, 0.1).unwrap();
    /// ```
    pub fn raw_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> XResult<f64> {
        let traj = self.duration(duration).unwrap();
        traj.raw_moment(order, particles, time_step)
    }

    /// Get the central moment of the Brownian bridge simulation
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the Brownian bridge simulation.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the Brownian bridge simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the central moment of the Brownian bridge simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let bb = BrownianBridge;
    /// let msd = bb.msd(1.0, 1000, 0.1).unwrap();
    /// ```
    pub fn central_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> XResult<f64> {
        let traj = self.duration(duration).unwrap();
        traj.central_moment(order, particles, time_step)
    }

    /// Get the first passage time of the Brownian bridge simulation
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the Brownian bridge simulation.
    /// * `max_duration` - The maximum duration of the Brownian bridge simulation.
    /// * `time_step` - The time step of the Brownian bridge simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the first passage time of the Brownian bridge simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let bb = BrownianBridge;
    /// let fpt = bm.fpt((-1.0, 1.0), 1000.0, 0.1).unwrap();
    /// ```
    pub fn fpt(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        max_duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<Option<f64>> {
        let fpt = FirstPassageTime::new(self, domain)?;
        fpt.simulate(max_duration, time_step)
    }

    /// Get the occupation time of the Brownian bridge simulation
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the Brownian bridge simulation.
    /// * `duration` - The duration of the Brownian bridge simulation.
    /// * `time_step` - The time step of the Brownian bridge simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the occupation time of the Brownian bridge simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let bb = BrownianBridge;
    /// let ot = bm.occupation_time((-1.0, 1.0), 1000.0, 0.1).unwrap();
    /// ```
    pub fn occupation_time(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<f64> {
        let ot = OccupationTime::new(self, domain, duration)?;
        ot.simulate(time_step)
    }
}

/// impl `ContinuousProcess` trait for Brownian motion
impl ContinuousProcess for BrownianBridge {
    /// Simulate Brownian bridge
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the Brownian bridge simulation.
    /// * `time_step` - The time step of the Brownian bridge simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the Brownian bridge simulation.
    ///
    /// # Example
    ///
    /// ```rust
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
/// This function simulates Brownian bridge.
///
/// # Arguments
///
/// * `duration` - The duration of the Brownian bridge.
/// * `time_step` - The time step of the Brownian bridge.
///
/// # Returns
///
/// A tuple containing the time and the position of the Brownian bridge simulation.
///
/// # Example
///
/// ```rust
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
    use crate::simulation::prelude::{ContinuousTrajectoryTrait, Moment};

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
