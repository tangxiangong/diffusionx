use crate::{SimulationError, XResult};
use rayon::prelude::*;

pub type Pair = (Vec<f64>, Vec<f64>);
pub type PointPair = (Vec<f64>, Vec<i64>);

/// Continuous process trait
pub trait ContinuousProcess: Clone + Send + Sync {
    /// Simulate the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the simulation.
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair>;
}

pub trait PointProcess: Clone + Send + Sync {
    /// Simulate the point process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the simulation.
    fn simulate_with_duration(&self, duration: impl Into<f64>) -> XResult<PointPair>;

    /// Simulate the point process with a given number of steps
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the simulation.
    fn simulate_with_step(&self, num_step: usize) -> XResult<PointPair>;
}

/// Continuous trajectory
///
/// # Fields
///
/// * `sp` - The simulation object.
/// * `duration` - The duration of the simulation.
pub struct ContinuousTrajectory<SP: ContinuousProcess> {
    pub(crate) sp: SP,
    pub(crate) duration: f64,
}

impl<SP: ContinuousProcess> ContinuousTrajectory<SP> {
    pub fn new(sp: SP, duration: impl Into<f64>) -> XResult<Self> {
        let duration = duration.into();
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(
                "duration must be positive".to_string(),
            )
            .into());
        }
        Ok(Self { sp, duration })
    }

    pub fn simulate(&self, time_step: f64) -> XResult<Pair> {
        self.sp.simulate(self.duration, time_step)
    }
}

/// Point process trajectory
///
/// # Fields
///
/// * `sp` - The simulation object.
/// * `duration` - The duration of the simulation.
pub struct PointTrajectory<SP: PointProcess> {
    pub(crate) sp: SP,
    pub(crate) duration: Option<f64>,
    pub(crate) num_step: Option<usize>,
}

impl<SP: PointProcess> PointTrajectory<SP> {
    pub fn with_duration(sp: SP, duration: impl Into<f64>) -> XResult<Self> {
        let duration = duration.into();
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(
                "duration must be positive".to_string(),
            )
            .into());
        }
        Ok(Self {
            sp,
            duration: Some(duration),
            num_step: None,
        })
    }

    pub fn with_step(sp: SP, num_step: usize) -> XResult<Self> {
        if num_step == 0 {
            return Err(SimulationError::InvalidParameters(
                "num_step must be positive".to_string(),
            )
            .into());
        }
        Ok(Self {
            sp,
            duration: None,
            num_step: Some(num_step),
        })
    }

    pub fn simulate_with_duration(&self) -> XResult<PointPair> {
        if self.duration.is_none() {
            return Err(SimulationError::InvalidParameters(
                "duration must be provided".to_string(),
            )
            .into());
        }
        self.sp.simulate_with_duration(self.duration.unwrap())
    }

    pub fn simulate_with_step(&self) -> XResult<PointPair> {
        if self.num_step.is_none() {
            return Err(SimulationError::InvalidParameters(
                "num_step must be provided".to_string(),
            )
            .into());
        }
        self.sp.simulate_with_step(self.num_step.unwrap())
    }
}

/// Trajectory trait
pub trait ContinuousTrajectoryTrait: ContinuousProcess {
    fn duration(&self, duration: impl Into<f64>) -> XResult<ContinuousTrajectory<Self>> {
        let traj = ContinuousTrajectory::new(self.clone(), duration)?;
        Ok(traj)
    }
}

impl<SP: ContinuousProcess> ContinuousTrajectoryTrait for SP {}

pub trait PointTrajectoryTrait: PointProcess {
    fn duration(&self, duration: impl Into<f64>) -> XResult<PointTrajectory<Self>> {
        let traj = PointTrajectory::with_duration(self.clone(), duration)?;
        Ok(traj)
    }

    fn step(&self, num_step: usize) -> XResult<PointTrajectory<Self>> {
        let traj = PointTrajectory::with_step(self.clone(), num_step)?;
        Ok(traj)
    }
}

impl<SP: PointProcess> PointTrajectoryTrait for SP {}

/// Moment trait
pub trait Moment {
    /// Get the raw moment of the simulation
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the raw moment of the simulation.
    fn raw_moment(&self, order: i32, particles: usize, time_step: f64) -> XResult<f64>;

    /// Get the central moment of the simulation
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the central moment of the simulation.
    fn central_moment(&self, order: i32, particles: usize, time_step: f64) -> XResult<f64>;
}

impl<SP: ContinuousProcess> Moment for ContinuousTrajectory<SP> {
    fn raw_moment(&self, order: i32, particles: usize, time_step: f64) -> XResult<f64> {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(
                "particles must be positive".to_string(),
            )
            .into());
        }
        if order < 0 {
            return Err(SimulationError::InvalidParameters(
                "order must be non-negative".to_string(),
            )
            .into());
        }
        if order == 0 {
            return Ok(0.0);
        }

        let sp = self.sp.clone();
        let duration = self.duration;

        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let (_, x) = sp.simulate(duration, time_step)?;
                let end_position = x.last();
                match end_position {
                    Some(position) => Ok(position.powi(order)),
                    None => Err(SimulationError::Unknown.into()),
                }
            })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / particles as f64;
        Ok(result)
    }
    fn central_moment(&self, order: i32, particles: usize, time_step: f64) -> XResult<f64> {
        let mean = self.raw_moment(order, particles, time_step)?;
        let sp = self.sp.clone();
        let duration = self.duration;
        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let (_, x) = sp.simulate(duration, time_step)?;
                let end_position = x.last();
                match end_position {
                    Some(position) => Ok((position - mean).powi(order)),
                    None => Err(SimulationError::Unknown.into()),
                }
            })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / particles as f64;
        Ok(result)
    }
}

impl<SP: PointProcess> Moment for PointTrajectory<SP> {
    fn raw_moment(&self, order: i32, particles: usize, _time_step: f64) -> XResult<f64> {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(
                "particles must be positive".to_string(),
            )
            .into());
        }
        if order < 0 {
            return Err(SimulationError::InvalidParameters(
                "order must be non-negative".to_string(),
            )
            .into());
        }
        if order == 0 {
            return Ok(0.0);
        }

        let sp = self.sp.clone();

        if self.duration.is_none() {
            return Err(SimulationError::InvalidParameters(
                "duration must be provided".to_string(),
            )
            .into());
        }
        let duration = self.duration.unwrap();

        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let (_, x) = sp.simulate_with_duration(duration)?;
                let end_position = x.last();
                match end_position {
                    Some(position) => Ok((*position as f64).powi(order)),
                    None => Err(SimulationError::Unknown.into()),
                }
            })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / particles as f64;
        Ok(result)
    }
    fn central_moment(&self, order: i32, particles: usize, _time_step: f64) -> XResult<f64> {
        let mean = self.raw_moment(order, particles, _time_step)?;
        let duration = self.duration.unwrap();
        let sp = self.sp.clone();
        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let (_, x) = sp.simulate_with_duration(duration)?;
                let end_position = x.last();
                match end_position {
                    Some(position) => Ok((*position as f64 - mean).powi(order)),
                    None => Err(SimulationError::Unknown.into()),
                }
            })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / particles as f64;
        Ok(result)
    }
}
