use crate::{IntExt, RealExt, SimulationError, XResult, simulation::prelude::Moment};

/// Discrete process trait
pub trait DiscreteProcess<T: RealExt = f64, U: IntExt = usize>: Send + Sync {
    /// Get the starting position
    fn start(&self) -> T;

    /// Get the ending position
    fn end(&self, num_step: U) -> XResult<T> {
        Ok(self.start() + self.displacement(num_step)?)
    }

    /// Get the displacement of the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    fn displacement(&self, num_step: U) -> XResult<T> {
        let x = self.simulate(num_step)?;
        let first_position = x.first();
        let end_position = x.last();
        match (first_position, end_position) {
            (Some(initial), Some(position)) => Ok(*position - *initial),
            _ => Err(SimulationError::Unknown.into()),
        }
    }

    /// Simulate the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    fn simulate(&self, num_step: U) -> XResult<Vec<T>>;

    /// Get the mean of the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    /// * `particles` - The number of particles.
    fn mean(&self, num_step: U, particles: usize) -> XResult<f64>
    where
        Self: DiscreteTrajectoryTrait<T, U>,
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
    fn msd(&self, num_step: U, particles: usize) -> XResult<f64>
    where
        Self: DiscreteTrajectoryTrait<T, U>,
    {
        let traj = self.step(num_step)?;
        traj.msd(particles, 0.1)
    }

    /// Get the raw moment of the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    fn raw_moment(&self, num_step: U, order: i32, particles: usize) -> XResult<f64>
    where
        Self: DiscreteTrajectoryTrait<T, U>,
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
    fn central_moment(&self, num_step: U, order: i32, particles: usize) -> XResult<f64>
    where
        Self: DiscreteTrajectoryTrait<T, U>,
    {
        let traj = self.step(num_step)?;
        traj.central_moment(order, particles, 0.1)
    }

    /// Get the fractional raw moment of the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    fn frac_raw_moment(&self, num_step: U, order: f64, particles: usize) -> XResult<f64>
    where
        Self: DiscreteTrajectoryTrait<T, U>,
    {
        let traj = self.step(num_step)?;
        traj.frac_raw_moment(order, particles, 0.1)
    }

    /// Get the fractional central moment of the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    fn frac_central_moment(&self, num_step: U, order: f64, particles: usize) -> XResult<f64>
    where
        Self: DiscreteTrajectoryTrait<T, U>,
    {
        let traj = self.step(num_step)?;
        traj.frac_central_moment(order, particles, 0.1)
    }
}

pub trait DiscreteTrajectoryTrait<T: RealExt = f64, U: IntExt = usize>:
    DiscreteProcess<T, U> + Clone
{
    /// Create a `DiscreteTrajectory` with given number of steps
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the trajectory
    fn step(&self, num_step: U) -> XResult<DiscreteTrajectory<Self, T, U>> {
        let traj = DiscreteTrajectory::new(self.clone(), num_step)?;
        Ok(traj)
    }
}

impl<SP: DiscreteProcess<T, U> + Clone, T: RealExt, U: IntExt> DiscreteTrajectoryTrait<T, U>
    for SP
{
}

/// Discrete trajectory
#[derive(Debug, Clone)]
pub struct DiscreteTrajectory<
    SP: DiscreteProcess<T, U> + Clone,
    T: RealExt = f64,
    U: IntExt = usize,
> {
    /// The discrete process
    pub(crate) sp: SP,
    /// The number of steps
    pub(crate) num_step: U,
    pub(crate) _marker: std::marker::PhantomData<T>,
}

impl<SP: DiscreteProcess<T, U> + Clone, T: RealExt, U: IntExt> DiscreteTrajectory<SP, T, U> {
    /// Create a new `DiscreteTrajetory` with given `DiscreteProcess` and num of steps.
    pub fn new(sp: SP, num_step: U) -> XResult<Self> {
        if num_step == U::zero() {
            return Err(SimulationError::InvalidParameters(
                "num_step must be positive".to_string(),
            )
            .into());
        }
        Ok(Self {
            sp,
            num_step,
            _marker: std::marker::PhantomData,
        })
    }

    /// Get the discrete process
    pub fn get_process(&self) -> &SP {
        &self.sp
    }

    /// Get the number of steps of the trajectory
    pub fn get_num_step(&self) -> U {
        self.num_step
    }

    /// Simulate method
    pub fn simulate(&self) -> XResult<Vec<T>> {
        self.sp.simulate(self.num_step)
    }
}
