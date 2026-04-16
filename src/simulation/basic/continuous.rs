use crate::{
    FloatExt, SimulationError, XResult,
    simulation::prelude::{FirstPassageTime, Moment, OccupationTime, TAMSD},
};

/// Common interface for continuous-time stochastic processes.
///
/// A continuous process returns sampled values `(t, x)` over a requested duration.
/// Implementors define the path sampler through [`ContinuousProcess::simulate`];
/// the default methods derive common observables such as displacement, first
/// passage time, occupation time, and ensemble moments from repeated simulations.
///
/// Conceptually, the displacement over a horizon \(T\) is
///
/// $$\Delta X(T) = X(T) - X(0).$$
pub trait ContinuousProcess<T: FloatExt = f64>: Send + Sync {
    /// Simulate the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `time_step` - The time step of the simulation.
    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)>;

    /// Get the displacement of the continuous process.
    ///
    /// The default implementation simulates one path and returns `X(duration) - X(0)`.
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `time_step` - The time step of the simulation.
    fn displacement(&self, duration: T, time_step: T) -> XResult<T> {
        let (_, x) = self.simulate(duration, time_step)?;
        match (x.first(), x.last()) {
            (Some(first), Some(last)) => Ok(*last - *first),
            _ => Err(SimulationError::Unknown.into()),
        }
    }

    /// Get the starting position.
    fn start(&self) -> T;

    /// Get the ending position at the requested duration.
    fn end(&self, duration: T, time_step: T) -> XResult<T> {
        let delta_x = self.displacement(duration, time_step)?;
        Ok(self.start() + delta_x)
    }

    /// Get the mean of the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    fn mean(&self, duration: T, particles: usize, time_step: T) -> XResult<T>
    where
        Self: ContinuousTrajectoryTrait<T>,
    {
        let traj = self.duration(duration)?;
        traj.mean(particles, time_step)
    }

    /// Get the mean square displacement of the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    fn msd(&self, duration: T, particles: usize, time_step: T) -> XResult<T>
    where
        Self: ContinuousTrajectoryTrait<T>,
    {
        let traj = self.duration(duration)?;
        traj.msd(particles, time_step)
    }

    /// Get the raw moment of the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the continuous process.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the continuous process.
    fn raw_moment(&self, duration: T, order: i32, particles: usize, time_step: T) -> XResult<T>
    where
        Self: ContinuousTrajectoryTrait<T>,
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
    fn central_moment(&self, duration: T, order: i32, particles: usize, time_step: T) -> XResult<T>
    where
        Self: ContinuousTrajectoryTrait<T>,
    {
        let traj = self.duration(duration)?;
        traj.central_moment(order, particles, time_step)
    }

    /// Get the fractional raw moment of the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the continuous process.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the continuous process.
    fn frac_raw_moment(&self, duration: T, order: T, particles: usize, time_step: T) -> XResult<T>
    where
        Self: ContinuousTrajectoryTrait<T>,
    {
        let traj = self.duration(duration)?;
        traj.frac_raw_moment(order, particles, time_step)
    }

    /// Get the fractional central moment of the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the continuous process.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the continuous process.
    fn frac_central_moment(
        &self,
        duration: T,
        order: T,
        particles: usize,
        time_step: T,
    ) -> XResult<T>
    where
        Self: ContinuousTrajectoryTrait<T>,
    {
        let traj = self.duration(duration)?;
        traj.frac_central_moment(order, particles, time_step)
    }

    /// Get the first passage time of the continuous process
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain which the first passage time is interested in.
    /// * `max_duration` - The maximum duration of the continuous process. If the process does not exit the domain before the maximum duration, the function returns None.
    /// * `time_step` - The time step of the simulation.
    fn fpt(&self, domain: (T, T), max_duration: T, time_step: T) -> XResult<Option<T>>
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
    fn occupation_time(&self, domain: (T, T), duration: T, time_step: T) -> XResult<T>
    where
        Self: Sized,
    {
        let ot = OccupationTime::new(self, domain, duration)?;
        ot.simulate(time_step)
    }

    /// Get the time-averaged mean square displacement of the continuous process.
    ///
    /// This estimates
    ///
    /// $$\overline{\delta^2(\Delta; T)}
    /// = \frac{1}{T-\Delta}\int_0^{T-\Delta}
    /// \left[X(t+\Delta)-X(t)\right]^2\,dt.$$
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `delta` - The lag length.
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    fn tamsd(&self, duration: T, delta: T, time_step: T, quad_order: usize) -> XResult<T>
    where
        Self: Sized,
    {
        let tamsd = TAMSD::new(self, duration, delta)?;
        tamsd.simulate(time_step, quad_order)
    }

    /// Get the ensemble average of the time-averaged mean square displacement of the continuous process.
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `delta` - The lag length.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    fn eatamsd(
        &self,
        duration: T,
        delta: T,
        particles: usize,
        time_step: T,
        quad_order: usize,
    ) -> XResult<T>
    where
        Self: Sized,
    {
        let tamsd = TAMSD::new(self, duration, delta)?;
        tamsd.mean(particles, time_step, quad_order)
    }
}

/// Continuous trajectory
#[derive(Debug, Clone)]
pub struct ContinuousTrajectory<SP, T: FloatExt = f64>
where
    SP: ContinuousProcess<T> + Clone,
{
    /// The continuous process
    pub(crate) sp: SP,
    /// The duration of the simulation
    pub(crate) duration: T,
}

/// Extension trait for binding a continuous process to a fixed simulation duration.
///
/// This trait is implemented for every cloneable [`ContinuousProcess`]. It provides
/// the ergonomic `process.duration(t)` constructor used by moment estimators.
pub trait ContinuousTrajectoryTrait<T: FloatExt>: ContinuousProcess<T> + Clone {
    /// Create a [`ContinuousTrajectory`] over `duration_arg`.
    ///
    /// The returned trajectory keeps a clone of the process and reuses the fixed
    /// duration for subsequent simulations and ensemble statistics.
    fn duration(&self, duration_arg: T) -> XResult<ContinuousTrajectory<Self, T>> {
        let traj = ContinuousTrajectory::new(self.clone(), duration_arg)?;
        Ok(traj)
    }
}

impl<T: FloatExt, SP: ContinuousProcess<T> + Clone> ContinuousTrajectoryTrait<T> for SP {}

impl<T: FloatExt, SP: ContinuousProcess<T> + Clone> ContinuousTrajectory<SP, T> {
    /// Create a new `ContinuousTrajectory` with given `ContinuousProcess` and duration.
    ///
    /// # Arguments
    ///
    /// * `sp` - The continuous process.
    /// * `duration` - The duration of the simulation.
    pub fn new(sp: SP, duration: T) -> XResult<Self> {
        if duration <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {duration:?}"
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
    pub fn get_duration(&self) -> T {
        self.duration
    }

    /// Simulate method
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the simulation.
    pub fn simulate(&self, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        self.sp.simulate(self.duration, time_step)
    }
}
