use crate::{SimulationError, XResult};

use super::{DiscretePair, Moment};

/// Discrete process trait
pub trait DiscreteProcess: Send + Sync + Sized {
    /// Simulate the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    fn simulate(&self, num_step: usize) -> XResult<DiscretePair>;

    /// Get the mean of the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    /// * `particles` - The number of particles.
    fn mean(&self, num_step: usize, particles: usize) -> XResult<f64> {
        let traj = self.step(num_step)?;
        traj.raw_moment(1, particles, 0.1)
    }

    /// Get the mean square displacement of the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    /// * `particles` - The number of particles.
    fn msd(&self, num_step: usize, particles: usize) -> XResult<f64> {
        let traj = self.step(num_step)?;
        traj.central_moment(2, particles, 0.1)
    }

    /// Get the raw moment of the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    fn raw_moment(&self, num_step: usize, order: i32, particles: usize) -> XResult<f64> {
        let traj = self.step(num_step)?;
        traj.raw_moment(order, particles, 0.1)
    }

    /// Get the central moment of the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    fn central_moment(&self, num_step: usize, order: i32, particles: usize) -> XResult<f64> {
        let traj = self.step(num_step)?;
        traj.central_moment(order, particles, 0.1)
    }
}

pub trait DiscreteTrajectoryTrait: DiscreteProcess {
    /// Create a `DiscreteTrajectory` with given number of steps
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the trajectory
    fn step(&self, num_step: usize) -> XResult<DiscreteTrajectory<Self>> {
        let traj = DiscreteTrajectory::new(self, num_step)?;
        Ok(traj)
    }
}

impl<SP: DiscreteProcess> DiscreteTrajectoryTrait for SP {}

/// Discrete trajectory
#[derive(Debug, Clone)]
pub struct DiscreteTrajectory<'a, SP: DiscreteProcess> {
    /// The discrete process
    pub(crate) sp: &'a SP,
    /// The number of steps
    pub(crate) num_step: usize,
}

impl<'a, SP: DiscreteProcess> DiscreteTrajectory<'a, SP> {
    /// Create a new `DiscreteTrajetory` with given `DiscreteProcess` and num of steps.
    pub fn new(sp: &'a SP, num_step: usize) -> XResult<Self> {
        if num_step == 0 {
            return Err(SimulationError::InvalidParameters(
                "num_step must be positive".to_string(),
            )
            .into());
        }
        Ok(Self { sp, num_step })
    }

    /// Get the discrete process
    pub fn get_process(&self) -> &'a SP {
        self.sp
    }

    /// Get the number of steps of the trajectory
    pub fn get_num_step(&self) -> usize {
        self.num_step
    }

    /// Simulate method
    pub fn simulate(&self) -> XResult<DiscretePair> {
        self.sp.simulate(self.num_step)
    }
}
