use crate::{
    SimulationError, XResult,
    simulation::prelude::{FirstPassageTime, OccupationTime},
};

use super::{Moment, Pair};
use std::sync::Arc;

/// Point process trait
pub trait PointProcess: Send + Sync {
    /// Simulate the point process with given duration
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    fn simulate_with_duration(&self, duration: f64) -> XResult<Pair>
    where
        Self: Sized, // simulate_with_step is called, which is dyn-dispatchable
    {
        let mut num_step = duration.ceil() as usize;
        let (t, x) = loop {
            let (t, x) = self.simulate_with_step(num_step)?;
            if t.last().is_none() {
                return Err(SimulationError::Unknown.into());
            }
            let end_time = *t.last().unwrap();
            if end_time >= duration {
                break (t, x);
            }
            num_step *= 2;
        };
        let index = t.iter().position(|&time| time >= duration).unwrap();
        let mut t_ = vec![0.0; index + 1];
        let mut x_ = vec![0.0; index + 1];
        t_[..index].copy_from_slice(&t[..index]);
        x_[..index].copy_from_slice(&x[..index]);
        if t[index] > duration {
            t_[index] = duration;
            x_[index] = x_[index - 1];
        } else {
            t_[index] = t[index];
            x_[index] = x[index];
        }

        Ok((t_, x_))
    }

    /// Get the mean of the point process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `particles` - The number of particles.
    fn mean(&self, duration: f64, particles: usize) -> XResult<f64>
    where
        Self: Sized + Clone + PointTrajectoryTrait,
    {
        let traj = self.duration(duration)?;
        traj.raw_moment(1, particles, 0.1)
    }

    /// Get the mean square displacement of the point process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `particles` - The number of particles.
    fn msd(&self, duration: f64, particles: usize) -> XResult<f64>
    where
        Self: Sized + Clone + PointTrajectoryTrait,
    {
        let traj = self.duration(duration)?;
        traj.central_moment(2, particles, 0.1)
    }

    /// Get the raw moment of the point process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    fn raw_moment(&self, duration: f64, order: i32, particles: usize) -> XResult<f64>
    where
        Self: Sized + Clone + PointTrajectoryTrait,
    {
        let traj = self.duration(duration)?;
        traj.raw_moment(order, particles, 0.1)
    }

    /// Get the central moment of the point process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    fn central_moment(&self, duration: f64, order: i32, particles: usize) -> XResult<f64>
    where
        Self: Sized + Clone + PointTrajectoryTrait,
    {
        let traj = self.duration(duration)?;
        traj.central_moment(order, particles, 0.1)
    }

    /// Get the first passage time of the point process
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain which the first passage time is interested in.
    /// * `max_duration` - The maximum duration of the simulation. If the process does not exit the domain before the maximum duration, the function returns None.
    fn fpt(&self, domain: (f64, f64), max_duration: f64) -> XResult<Option<f64>>
    where
        Self: Sized,
    {
        let fpt = FirstPassageTime::new(self, domain)?;
        fpt.simulate_p(max_duration)
    }

    /// Get the occupation time of the point process
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain which the occupation time is interested in.
    /// * `duration` - The duration of the simulation.
    fn occupation_time(&self, domain: (f64, f64), duration: f64) -> XResult<f64>
    where
        Self: Sized,
    {
        let ot = OccupationTime::new(self, domain, duration)?;
        ot.simulate_p()
    }

    /// Simulate the point process with a given number of steps
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    fn simulate_with_step(&self, num_step: usize) -> XResult<Pair>;
}

/// Point process trajectory
#[derive(Debug, Clone)]
pub struct PointTrajectory<SP: PointProcess> {
    /// The point process
    pub(crate) sp: Arc<SP>,
    /// The duration of the trajectory
    pub(crate) duration: Option<f64>,
    /// The number of steps of the trajectory
    pub(crate) num_step: Option<usize>,
}

pub trait PointTrajectoryTrait: PointProcess
where
    Self: Sized + Clone,
{
    /// Create a `PointTrajectory` with given duration
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the trajectory
    fn duration(&self, duration: f64) -> XResult<PointTrajectory<Self>> {
        let traj = PointTrajectory::with_duration(self.clone(), duration)?;
        Ok(traj)
    }

    /// Create a `PointTrajectory` with given number of steps
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the trajectory
    fn step(&self, num_step: usize) -> XResult<PointTrajectory<Self>> {
        let traj = PointTrajectory::with_step(self.clone(), num_step)?;
        Ok(traj)
    }
}

impl<SP: PointProcess + Sized + Clone> PointTrajectoryTrait for SP {}

impl<SP: PointProcess> PointTrajectory<SP> {
    /// Get the point process
    pub fn get_process(&self) -> &SP {
        &self.sp
    }

    /// Get the duration of the trajectory
    pub fn get_duration(&self) -> Option<f64> {
        self.duration
    }

    /// Get the number of steps of the trajectory
    pub fn get_num_step(&self) -> Option<usize> {
        self.num_step
    }

    /// Create a `PointTrajectory` with duration.
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the trajectory
    pub fn with_duration(sp: SP, duration: f64) -> XResult<Self> {
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {}",
                duration
            ))
            .into());
        }
        Ok(Self {
            sp: Arc::new(sp),
            duration: Some(duration),
            num_step: None,
        })
    }

    /// Create a `PointTrajectory` with num of steps.
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the trajectory
    pub fn with_step(sp: SP, num_step: usize) -> XResult<Self> {
        if num_step == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `num_step` must be positive, got {}",
                num_step
            ))
            .into());
        }
        Ok(Self {
            sp: Arc::new(sp),
            duration: None,
            num_step: Some(num_step),
        })
    }

    /// Simulate the trajectory with duration
    pub fn simulate_with_duration(&self) -> XResult<Pair> {
        if self.duration.is_none() {
            return Err(SimulationError::InvalidParameters(
                "The `duration` must be provided".to_string(),
            )
            .into());
        }
        let duration = self.duration.unwrap();
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {}",
                duration
            ))
            .into());
        }
        self.sp.simulate_with_duration(duration)
    }

    /// Simulate with number of steps
    pub fn simulate_with_step(&self) -> XResult<Pair> {
        if self.num_step.is_none() {
            return Err(SimulationError::InvalidParameters(
                "num_step must be provided".to_string(),
            )
            .into());
        }
        let num_step = self.num_step.unwrap();
        if num_step == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `num_step` must be positive, got {}",
                num_step
            ))
            .into());
        }
        self.sp.simulate_with_step(num_step)
    }
}
