use crate::{SimulationError, XResult};

use super::{DiscretePair, Moment};

/// Discrete process trait
pub trait DiscreteProcess: Clone + Send + Sync {
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

    /// Create a `DiscreteTrajectory` with given number of steps
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the trajectory
    fn step(&self, num_step: usize) -> XResult<DiscreteTrajectory<Self>> {
        let traj = DiscreteTrajectory::new(self.clone(), num_step)?;
        Ok(traj)
    }
}

/// Discrete trajectory
pub struct DiscreteTrajectory<SP: DiscreteProcess> {
    /// The discrete process
    pub(crate) sp: SP,
    /// The number of steps
    pub(crate) num_step: usize,
}

impl<SP: DiscreteProcess> DiscreteTrajectory<SP> {
    /// Create a new `DiscreteTrajetory` with given `DiscreteProcess` and num of steps.
    pub fn new(sp: SP, num_step: usize) -> XResult<Self> {
        if num_step == 0 {
            return Err(SimulationError::InvalidParameters(
                "num_step must be positive".to_string(),
            )
            .into());
        }
        Ok(Self { sp, num_step })
    }

    /// Get the discrete process
    pub fn get_process(&self) -> &SP {
        &self.sp
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
