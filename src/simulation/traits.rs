//! Traits for stochastic processes
//!
//! - Continuous process [ContinuousProcess]
//! - Point process [PointProcess]
//! - Moment [Moment]
//!

use crate::{SimulationError, XResult};
use rayon::prelude::*;

pub type Pair = (Vec<f64>, Vec<f64>);
pub type DiscretePair = (Vec<usize>, Vec<f64>);

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

/// Discrete process trait
pub trait DiscreteProcess: Clone + Send + Sync {
    /// Simulate the discrete process
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the simulation.
    fn simulate(&self, num_step: usize) -> XResult<DiscretePair>;
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
    fn simulate_with_duration(&self, duration: impl Into<f64>) -> XResult<Pair> {
        let duration = duration.into();
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

    /// Simulate the point process with a given number of steps
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the simulation.
    fn simulate_with_step(&self, num_step: usize) -> XResult<Pair>;
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

/// Discrete trajectory
///
/// # Fields
///
/// * `sp` - The simulation object.
/// * `num_step` - The number of steps of the simulation.
pub struct DiscreteTrajectory<SP: DiscreteProcess> {
    pub(crate) sp: SP,
    pub(crate) num_step: usize,
}

impl<SP: DiscreteProcess> DiscreteTrajectory<SP> {
    pub fn new(sp: SP, num_step: usize) -> XResult<Self> {
        if num_step == 0 {
            return Err(SimulationError::InvalidParameters(
                "num_step must be positive".to_string(),
            )
            .into());
        }
        Ok(Self { sp, num_step })
    }

    pub fn simulate(&self) -> XResult<DiscretePair> {
        self.sp.simulate(self.num_step)
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

    pub fn simulate_with_duration(&self) -> XResult<Pair> {
        if self.duration.is_none() {
            return Err(SimulationError::InvalidParameters(
                "duration must be provided".to_string(),
            )
            .into());
        }
        self.sp.simulate_with_duration(self.duration.unwrap())
    }

    pub fn simulate_with_step(&self) -> XResult<Pair> {
        if self.num_step.is_none() {
            return Err(SimulationError::InvalidParameters(
                "num_step must be provided".to_string(),
            )
            .into());
        }
        self.sp.simulate_with_step(self.num_step.unwrap())
    }
}

/// Continuous trajectory trait
pub trait ContinuousTrajectoryTrait: ContinuousProcess {
    fn duration(&self, duration: impl Into<f64>) -> XResult<ContinuousTrajectory<Self>> {
        let traj = ContinuousTrajectory::new(self.clone(), duration)?;
        Ok(traj)
    }
}

impl<SP: ContinuousProcess> ContinuousTrajectoryTrait for SP {}

/// Discrete trajectory trait
pub trait DiscreteTrajectoryTrait: DiscreteProcess {
    fn step(&self, num_step: usize) -> XResult<DiscreteTrajectory<Self>> {
        let traj = DiscreteTrajectory::new(self.clone(), num_step)?;
        Ok(traj)
    }
}

impl<SP: DiscreteProcess> DiscreteTrajectoryTrait for SP {}

/// Point trajectory trait
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

impl<SP: DiscreteProcess> Moment for DiscreteTrajectory<SP> {
    fn raw_moment(&self, order: i32, particles: usize, _: f64) -> XResult<f64> {
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
        let num_step = self.num_step;

        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let (_, x) = sp.simulate(num_step)?;
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
    fn central_moment(&self, order: i32, particles: usize, _: f64) -> XResult<f64> {
        let mean = self.raw_moment(order, particles, 0.01)?;
        let sp = self.sp.clone();
        let num_step = self.num_step;
        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let (_, x) = sp.simulate(num_step)?;
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
                    Some(position) => Ok(position.powi(order)),
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
