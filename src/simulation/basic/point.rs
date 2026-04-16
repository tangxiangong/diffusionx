use crate::{
    FloatExt, RealExt, SimulationError, XResult,
    simulation::prelude::{FirstPassageTime, Moment, OccupationTime},
};

/// Common interface for point processes.
///
/// `T` is the time type, and `X` is the state type. Point processes are sampled
/// at random event times and can be truncated either by duration or by event count.
pub trait PointProcess<T: FloatExt = f64, X: RealExt = T>: Send + Sync {
    /// Get the starting position.
    fn start(&self) -> X;

    /// Get the ending position at the requested duration.
    fn end(&self, duration: T) -> XResult<X> {
        Ok(self.start() + self.displacement(duration)?)
    }

    /// Get the displacement of the point process.
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    fn displacement(&self, duration: T) -> XResult<X> {
        if duration <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {duration:?}"
            ))
            .into());
        }
        let mut num_step = duration.ceil().to_usize().unwrap();
        let (t, x) = loop {
            let (t, x) = self.simulate_with_step(num_step)?;
            if t.last().is_none() {
                return Err(SimulationError::Unknown.into());
            }
            let end_time = *t.last().unwrap();
            if T::from(end_time).unwrap() >= duration {
                break (t, x);
            }
            num_step *= 2;
        };
        let index = t.iter().position(|&time| time >= duration).unwrap();

        let delta_x = if t[index] > duration {
            x[index - 1] - x[0]
        } else {
            x[index] - x[0]
        };

        Ok(delta_x)
    }
    /// Simulate the point process up to a fixed duration.
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    fn simulate_with_duration(&self, duration: T) -> XResult<(Vec<T>, Vec<X>)>
    where
        Self: Sized,
    {
        if duration <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {duration:?}"
            ))
            .into());
        }
        let mut num_step = duration.ceil().to_usize().unwrap();
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
        let mut t_ = vec![T::zero(); index + 1];
        let mut x_ = vec![X::zero(); index + 1];
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
    fn mean(&self, duration: T, particles: usize) -> XResult<f64>
    where
        Self: Sized + Clone + PointTrajectoryTrait<T, X>,
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
    fn msd(&self, duration: T, particles: usize) -> XResult<f64>
    where
        Self: Sized + Clone + PointTrajectoryTrait<T, X>,
    {
        let traj = self.duration(duration)?;
        traj.msd(particles, 0.1)
    }

    /// Get the raw moment of the point process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    fn raw_moment(&self, duration: T, order: i32, particles: usize) -> XResult<f64>
    where
        Self: Sized + Clone + PointTrajectoryTrait<T, X>,
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
    fn central_moment(&self, duration: T, order: i32, particles: usize) -> XResult<f64>
    where
        Self: Sized + Clone + PointTrajectoryTrait<T, X>,
    {
        let traj = self.duration(duration)?;
        traj.central_moment(order, particles, 0.1)
    }

    /// Get the fractional raw moment of the point process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    fn frac_raw_moment(&self, duration: T, order: f64, particles: usize) -> XResult<f64>
    where
        Self: Sized + Clone + PointTrajectoryTrait<T, X>,
    {
        let traj = self.duration(duration)?;
        traj.frac_raw_moment(order, particles, 0.1)
    }

    /// Get the fractional central moment of the point process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    fn frac_central_moment(&self, duration: T, order: f64, particles: usize) -> XResult<f64>
    where
        Self: Sized + Clone + PointTrajectoryTrait<T, X>,
    {
        let traj = self.duration(duration)?;
        traj.frac_central_moment(order, particles, 0.1)
    }

    /// Get the first passage time of the point process
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain which the first passage time is interested in.
    /// * `max_duration` - The maximum duration of the simulation. If the process does not exit the domain before the maximum duration, the function returns None.
    fn fpt(&self, domain: (X, X), max_duration: T) -> XResult<Option<T>>
    where
        Self: Sized + Clone,
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
    fn occupation_time(&self, domain: (X, X), duration: T) -> XResult<T>
    where
        Self: Sized + Clone,
    {
        let ot = OccupationTime::new(self, domain, duration)?;
        ot.simulate_p()
    }

    /// Simulate the point process with a given number of steps
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    fn simulate_with_step(&self, num_step: usize) -> XResult<(Vec<T>, Vec<X>)>;
}

/// Point process trajectory
#[derive(Debug, Clone)]
pub struct PointTrajectory<SP: PointProcess<T, X> + Clone, T: FloatExt = f64, X: RealExt = T> {
    /// The point process
    pub(crate) sp: SP,
    /// The duration of the trajectory
    pub(crate) duration: Option<T>,
    /// The number of steps of the trajectory
    pub(crate) num_step: Option<usize>,
    _marker: std::marker::PhantomData<X>,
}

/// Extension trait for binding a point process to either a duration or step count.
///
/// This trait is implemented for every cloneable [`PointProcess`]. It provides
/// the ergonomic `process.duration(t)` and `process.step(n)` constructors used by
/// moment estimators.
pub trait PointTrajectoryTrait<T: FloatExt = f64, X: RealExt = T>:
    PointProcess<T, X> + Clone
{
    /// Create a `PointTrajectory` with given duration
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the trajectory
    fn duration(&self, duration: T) -> XResult<PointTrajectory<Self, T, X>> {
        let traj = PointTrajectory::with_duration(self.clone(), duration)?;
        Ok(traj)
    }

    /// Create a `PointTrajectory` with given number of steps
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the trajectory
    fn step(&self, num_step: usize) -> XResult<PointTrajectory<Self, T, X>> {
        let traj = PointTrajectory::with_step(self.clone(), num_step)?;
        Ok(traj)
    }
}

impl<SP: PointProcess<T, X> + Sized + Clone, T: FloatExt, X: RealExt> PointTrajectoryTrait<T, X>
    for SP
{
}

impl<SP: PointProcess<T, X> + Clone, T: FloatExt, X: RealExt> PointTrajectory<SP, T, X> {
    /// Get the point process
    pub fn get_process(&self) -> &SP {
        &self.sp
    }

    /// Get the duration of the trajectory
    pub fn get_duration(&self) -> Option<T> {
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
    pub fn with_duration(sp: SP, duration: T) -> XResult<Self> {
        if duration <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {duration:?}"
            ))
            .into());
        }
        Ok(Self {
            sp: sp.clone(),
            duration: Some(duration),
            num_step: None,
            _marker: std::marker::PhantomData,
        })
    }

    /// Create a `PointTrajectory` with a fixed number of steps.
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the trajectory
    pub fn with_step(sp: SP, num_step: usize) -> XResult<Self> {
        if num_step == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `num_step` must be positive, got {num_step}"
            ))
            .into());
        }
        Ok(Self {
            sp: sp.clone(),
            duration: None,
            num_step: Some(num_step),
            _marker: std::marker::PhantomData,
        })
    }

    /// Simulate the trajectory over its stored duration.
    pub fn simulate_with_duration(&self) -> XResult<(Vec<T>, Vec<X>)> {
        if self.duration.is_none() {
            return Err(SimulationError::InvalidParameters(
                "The `duration` must be provided".to_string(),
            )
            .into());
        }
        let duration = self.duration.unwrap();
        if duration <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {duration:?}"
            ))
            .into());
        }
        self.sp.simulate_with_duration(duration)
    }

    /// Simulate the trajectory for its stored number of steps.
    pub fn simulate_with_step(&self) -> XResult<(Vec<T>, Vec<X>)> {
        if self.num_step.is_none() {
            return Err(SimulationError::InvalidParameters(
                "num_step must be provided".to_string(),
            )
            .into());
        }
        let num_step = self.num_step.unwrap();
        if num_step == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `num_step` must be positive, got {num_step}"
            ))
            .into());
        }
        self.sp.simulate_with_step(num_step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct UnitPoint;

    impl PointProcess<f64, f64> for UnitPoint {
        fn start(&self) -> f64 {
            0.0
        }

        fn simulate_with_step(&self, num_step: usize) -> XResult<(Vec<f64>, Vec<f64>)> {
            let t = (0..=num_step).map(|i| i as f64).collect::<Vec<_>>();
            let x = vec![0.0; num_step + 1];
            Ok((t, x))
        }
    }

    #[test]
    fn test_displacement_rejects_nonpositive_duration() {
        let process = UnitPoint;
        let result = std::panic::catch_unwind(|| process.displacement(-1.0));
        assert!(matches!(result, Ok(Err(_))));
    }

    #[test]
    fn test_simulate_with_duration_rejects_nonpositive_duration() {
        let process = UnitPoint;
        let result = process.simulate_with_duration(0.0);
        assert!(result.is_err());
    }
}
