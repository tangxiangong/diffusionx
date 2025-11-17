use super::{DiscretePair, Moment};
use crate::{SimulationError, XResult};
use rayon::prelude::*;

/// Discrete process trait
pub trait DiscreteProcess: Send + Sync {
    /// Get the displacement of the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    fn displacement(&self, num_step: usize) -> XResult<f64> {
        let (_, x) = self.simulate(num_step)?;
        let first_position = x.first();
        let end_position = x.last();
        match (first_position, end_position) {
            (Some(initial), Some(position)) => Ok(position - initial),
            _ => Err(SimulationError::Unknown.into()),
        }
    }

    /// Simulate the discrete process without checking the arguments
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    fn simulate_unchecked(&self, num_step: usize) -> XResult<DiscretePair>;

    /// Simulate the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    fn simulate(&self, num_step: usize) -> XResult<DiscretePair> {
        if num_step == 0 {
            return Ok((vec![], vec![]));
        }
        self.simulate_unchecked(num_step)
    }

    /// Get the mean of the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    /// * `particles` - The number of particles.
    fn mean(&self, num_step: usize, particles: usize) -> XResult<f64>
    where
        Self: DiscreteTrajectoryTrait,
    {
        let traj = self.step(num_step)?;
        traj.raw_moment(1, particles, 0.1)
    }

    /// Get the mean square displacement of the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    /// * `particles` - The number of particles.
    fn msd(&self, num_step: usize, particles: usize) -> XResult<f64>
    where
        Self: DiscreteTrajectoryTrait,
    {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {particles}"
            ))
            .into());
        }

        let values: Vec<f64> = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let (_, x) = self.simulate(num_step)?;
                let first_position = x.first();
                let end_position = x.last();
                match (first_position, end_position) {
                    (Some(initial), Some(position)) => {
                        Ok((position - initial) * (position - initial))
                    }
                    _ => Err(SimulationError::Unknown.into()),
                }
            })
            .collect::<XResult<Vec<_>>>()?;

        let result = values.iter().sum::<f64>() / particles as f64;
        Ok(result)
    }

    /// Get the raw moment of the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    fn raw_moment(&self, num_step: usize, order: i32, particles: usize) -> XResult<f64>
    where
        Self: DiscreteTrajectoryTrait,
    {
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
    fn central_moment(&self, num_step: usize, order: i32, particles: usize) -> XResult<f64>
    where
        Self: DiscreteTrajectoryTrait,
    {
        let traj = self.step(num_step)?;
        traj.central_moment(order, particles, 0.1)
    }
}

pub trait DiscreteTrajectoryTrait: DiscreteProcess + Clone {
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

impl<SP: DiscreteProcess + Clone> DiscreteTrajectoryTrait for SP {}

/// Discrete trajectory
#[derive(Debug, Clone)]
pub struct DiscreteTrajectory<SP: DiscreteProcess + Clone> {
    /// The discrete process
    pub(crate) sp: SP,
    /// The number of steps
    pub(crate) num_step: usize,
}

impl<SP: DiscreteProcess + Clone> DiscreteTrajectory<SP> {
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
