use crate::{
    SimulationError, XResult,
    simulation::prelude::{FirstPassageTime, OccupationTime},
};

use super::{Moment, Pair, TAMSD};

/// Continuous process trait
pub trait ContinuousProcess: Clone + Send + Sync {
    /// Simulate the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `time_step` - The time step of the simulation.
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair>;

    /// Get the mean of the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    fn mean(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.raw_moment(1, particles, time_step)
    }

    /// Get the mean square displacement of the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    fn msd(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.central_moment(2, particles, time_step)
    }

    /// Get the raw moment of the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the continuous process.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the continuous process.
    fn raw_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.raw_moment(order, particles, time_step)
    }

    /// Get the central moment of the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the continuous process.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the continuous process.
    fn central_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.central_moment(order, particles, time_step)
    }

    /// Get the first passage time of the continuous process
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain which the first passage time is interested in.
    /// * `max_duration` - The maximum duration of the continuous process. If the process does not exit the domain before the maximum duration, the function returns None.
    /// * `time_step` - The time step of the simulation.
    fn fpt(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        max_duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<Option<f64>> {
        let fpt = FirstPassageTime::new(self, domain)?;
        fpt.simulate(max_duration, time_step)
    }

    /// Get the occupation time of the continuous process
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain which the occupation time is interested in.
    /// * `duration` - The duration of the continuous process.
    /// * `time_step` - The time step of the simulation.
    fn occupation_time(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<f64> {
        let ot = OccupationTime::new(self, domain, duration)?;
        ot.simulate(time_step)
    }

    /// Get the time-averaged mean square displacement of the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `delta` - The slag length.
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    fn tamsd(
        &self,
        duration: impl Into<f64>,
        delta: impl Into<f64>,
        time_step: f64,
        quad_order: usize,
    ) -> XResult<f64> {
        let tamsd = TAMSD::new(self, duration, delta)?;
        tamsd.simulate(time_step, quad_order)
    }

    /// Get the ensemble average of the time-averaged mean square displacement of the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `delta` - The slag length.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    fn eatamsd(
        &self,
        duration: impl Into<f64>,
        delta: impl Into<f64>,
        particles: usize,
        time_step: f64,
        quad_order: usize,
    ) -> XResult<f64> {
        let tamsd = TAMSD::new(self, duration, delta)?;
        tamsd.mean(particles, time_step, quad_order)
    }

    /// Create a `ContinuousTrajectory` with given duration
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the trajectory
    fn duration(&self, duration: impl Into<f64>) -> XResult<ContinuousTrajectory<Self>> {
        let traj = ContinuousTrajectory::new(self.clone(), duration)?;
        Ok(traj)
    }
}

/// Continuous trajectory
pub struct ContinuousTrajectory<SP: ContinuousProcess> {
    /// The continuous process
    pub(crate) sp: SP,
    /// The duration of the simulation
    pub(crate) duration: f64,
}

impl<SP: ContinuousProcess> ContinuousTrajectory<SP> {
    /// Create a new `ContinuousTrajectory` with given `ContinuousProcess` and duration.
    ///
    /// # Arguments
    ///
    /// * `sp` - The continuous process.
    /// * `duration` - The duration of the simulation.
    pub fn new(sp: SP, duration: impl Into<f64>) -> XResult<Self> {
        let duration = duration.into();
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {}",
                duration
            ))
            .into());
        }
        Ok(Self { sp, duration })
    }

    /// Get the continuous process
    pub fn get_process(&self) -> &SP {
        &self.sp
    }

    /// Get the duration of the trajectory
    pub fn get_duration(&self) -> f64 {
        self.duration
    }

    /// Simulate method
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the simulation.
    pub fn simulate(&self, time_step: f64) -> XResult<Pair> {
        self.sp.simulate(self.duration, time_step)
    }
}
