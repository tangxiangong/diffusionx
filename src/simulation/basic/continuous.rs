use std::sync::Arc;

use crate::{
    SimulationError, XResult,
    simulation::prelude::{FirstPassageTime, OccupationTime},
};

use super::{Moment, Pair, TAMSD};

/// Continuous process trait
pub trait ContinuousProcess: Send + Sync {
    /// Simulate the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `time_step` - The time step of the simulation.
    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair>;

    /// Get the mean of the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    fn mean(&self, duration: f64, particles: usize, time_step: f64) -> XResult<f64>
    where
        Self: Sized + Clone + ContinuousTrajectoryTrait,
    {
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
    fn msd(&self, duration: f64, particles: usize, time_step: f64) -> XResult<f64>
    where
        Self: Sized + Clone + ContinuousTrajectoryTrait,
    {
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
        duration: f64,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> XResult<f64>
    where
        Self: Sized + Clone + ContinuousTrajectoryTrait,
    {
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
        duration: f64,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> XResult<f64>
    where
        Self: Sized + Clone + ContinuousTrajectoryTrait,
    {
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
    fn fpt(&self, domain: (f64, f64), max_duration: f64, time_step: f64) -> XResult<Option<f64>>
    where
        Self: Sized,
    {
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
    fn occupation_time(&self, domain: (f64, f64), duration: f64, time_step: f64) -> XResult<f64>
    where
        Self: Sized,
    {
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
    fn tamsd(&self, duration: f64, delta: f64, time_step: f64, quad_order: usize) -> XResult<f64>
    where
        Self: Sized,
    {
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
        duration: f64,
        delta: f64,
        particles: usize,
        time_step: f64,
        quad_order: usize,
    ) -> XResult<f64>
    where
        Self: Sized,
    {
        let tamsd = TAMSD::new(self, duration, delta)?;
        tamsd.mean(particles, time_step, quad_order)
    }
}

/// Continuous trajectory
#[derive(Debug, Clone)]
pub struct ContinuousTrajectory<SP: ContinuousProcess> {
    /// The continuous process
    pub(crate) sp: Arc<SP>,
    /// The duration of the simulation
    pub(crate) duration: f64,
}

pub trait ContinuousTrajectoryTrait: ContinuousProcess
where
    Self: Sized + Clone,
{
    fn duration(&self, duration_arg: f64) -> XResult<ContinuousTrajectory<Self>> {
        let traj = ContinuousTrajectory::new(self.clone(), duration_arg)?;
        Ok(traj)
    }
}

impl<SP: ContinuousProcess + Sized + Clone> ContinuousTrajectoryTrait for SP {}

impl<SP: ContinuousProcess> ContinuousTrajectory<SP> {
    /// Create a new `ContinuousTrajectory` with given `ContinuousProcess` and duration.
    ///
    /// # Arguments
    ///
    /// * `sp` - The continuous process.
    /// * `duration` - The duration of the simulation.
    pub fn new(sp: SP, duration: f64) -> XResult<Self> {
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {}",
                duration
            ))
            .into());
        }
        Ok(Self {
            sp: Arc::new(sp),
            duration,
        })
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
