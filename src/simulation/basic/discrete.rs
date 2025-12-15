use crate::{IntExt, RealExt, SimulationError, XResult, simulation::prelude::Moment};

/// Discrete process trait
pub trait DiscreteProcess<N: IntExt = usize, X: RealExt = f64>: Send + Sync {
    /// Get the starting position
    fn start(&self) -> X;

    /// Get the ending position
    fn end(&self, num_step: N) -> XResult<X> {
        Ok(self.start() + self.displacement(num_step)?)
    }

    /// Get the displacement of the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    fn displacement(&self, num_step: N) -> XResult<X> {
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
    fn simulate(&self, num_step: N) -> XResult<Vec<X>>;

    /// Get the mean of the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    /// * `particles` - The number of particles.
    fn mean(&self, num_step: N, particles: usize) -> XResult<f64>
    where
        Self: DiscreteTrajectoryTrait<N, X>,
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
    fn msd(&self, num_step: N, particles: usize) -> XResult<f64>
    where
        Self: DiscreteTrajectoryTrait<N, X>,
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
    fn raw_moment(&self, num_step: N, order: i32, particles: usize) -> XResult<f64>
    where
        Self: DiscreteTrajectoryTrait<N, X>,
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
    fn central_moment(&self, num_step: N, order: i32, particles: usize) -> XResult<f64>
    where
        Self: DiscreteTrajectoryTrait<N, X>,
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
    fn frac_raw_moment(&self, num_step: N, order: f64, particles: usize) -> XResult<f64>
    where
        Self: DiscreteTrajectoryTrait<N, X>,
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
    fn frac_central_moment(&self, num_step: N, order: f64, particles: usize) -> XResult<f64>
    where
        Self: DiscreteTrajectoryTrait<N, X>,
    {
        let traj = self.step(num_step)?;
        traj.frac_central_moment(order, particles, 0.1)
    }
}

pub trait DiscreteTrajectoryTrait<N: IntExt = usize, X: RealExt = f64>:
    DiscreteProcess<N, X> + Clone
{
    /// Create a `DiscreteTrajectory` with given number of steps
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the trajectory
    fn step(&self, num_step: N) -> XResult<DiscreteTrajectory<Self, N, X>> {
        let traj = DiscreteTrajectory::new(self.clone(), num_step)?;
        Ok(traj)
    }
}

impl<SP: DiscreteProcess<N, X> + Clone, N: IntExt, X: RealExt> DiscreteTrajectoryTrait<N, X>
    for SP
{
}

/// Discrete trajectory
#[derive(Debug, Clone)]
pub struct DiscreteTrajectory<
    SP: DiscreteProcess<N, X> + Clone,
    N: IntExt = usize,
    X: RealExt = f64,
> {
    /// The discrete process
    pub(crate) sp: SP,
    /// The number of steps
    pub(crate) num_step: N,
    pub(crate) _marker: std::marker::PhantomData<X>,
}

impl<SP: DiscreteProcess<N, X> + Clone, N: IntExt, X: RealExt> DiscreteTrajectory<SP, N, X> {
    /// Create a new `DiscreteTrajetory` with given `DiscreteProcess` and num of steps.
    pub fn new(sp: SP, num_step: N) -> XResult<Self> {
        if num_step == N::zero() {
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
    pub fn get_num_step(&self) -> N {
        self.num_step
    }

    /// Simulate method
    pub fn simulate(&self) -> XResult<Vec<X>> {
        self.sp.simulate(self.num_step)
    }
}
