//! Traits and structs for stochastic processes
//!
//! ## Traits
//! - Continuous process [ContinuousProcess]
//! - Point process [PointProcess]
//! - Discrete process [DiscreteProcess]
//! - Continuous trajectory [ContinuousTrajectory]
//! - Discrete trajectory [DiscreteTrajectory]
//! - Moment [Moment]
//! - Inverse process [Inverse]
//!
//! ## Structs
//! - ContinuousTrajectory [ContinuousTrajectory]
//! - DiscreteTrajectory [DiscreteTrajectory]
//! - PointTrajectory [PointTrajectory]
//! - TAMSD [TAMSD]
//!

use crate::{
    SimulationError, XResult,
    simulation::prelude::{FirstPassageTime, OccupationTime},
    utils::flatten_interpolate,
};
use gauss_quad::GaussLegendre;
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
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair>;

    /// Get the mean of the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    fn mean(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.raw_moment(1, particles, time_step)
    }

    /// Get the mean square displacement of the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    fn msd(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.central_moment(2, particles, time_step)
    }

    /// Get the raw moment of the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the continuous process.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the continuous process.
    fn raw_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> XResult<f64> {
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
    fn central_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.central_moment(order, particles, time_step)
    }

    /// Get the first passage time of the continuous process
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain which the first passage time is interested in.
    /// * `max_duration` - The maximum duration of the continuous process. If the process does not exit the domain before the maximum duration, the function returns None.
    /// * `time_step` - The time step of the simulation.
    fn fpt(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        max_duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<Option<f64>> {
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
    fn occupation_time(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<f64> {
        let ot = OccupationTime::new(self, domain, duration)?;
        ot.simulate(time_step)
    }

    /// Get the time-averaged mean square displacement of the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `delta` - The slag length.
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    fn tamsd(
        &self,
        duration: impl Into<f64>,
        delta: impl Into<f64>,
        time_step: f64,
        quad_order: usize,
    ) -> XResult<f64> {
        let tamsd = TAMSD::new(self, duration, delta)?;
        tamsd.simulate(time_step, quad_order)
    }

    /// Get the ensemble average of the time-averaged mean square displacement of the continuous process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `delta` - The slag length.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    fn eatamsd(
        &self,
        duration: impl Into<f64>,
        delta: impl Into<f64>,
        particles: usize,
        time_step: f64,
        quad_order: usize,
    ) -> XResult<f64> {
        let tamsd = TAMSD::new(self, duration, delta)?;
        tamsd.mean(particles, time_step, quad_order)
    }
}

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
}

/// Point process trait
pub trait PointProcess: Clone + Send + Sync {
    /// Simulate the point process with given duration
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
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

    /// Get the mean of the point process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `particles` - The number of particles.
    fn mean(&self, duration: impl Into<f64>, particles: usize) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.raw_moment(1, particles, 0.1)
    }

    /// Get the mean square displacement of the point process
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `particles` - The number of particles.
    fn msd(&self, duration: impl Into<f64>, particles: usize) -> XResult<f64> {
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
    fn raw_moment(&self, duration: impl Into<f64>, order: i32, particles: usize) -> XResult<f64> {
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
    fn central_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
    ) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.central_moment(order, particles, 0.1)
    }

    /// Get the first passage time of the point process
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain which the first passage time is interested in.
    /// * `max_duration` - The maximum duration of the simulation. If the process does not exit the domain before the maximum duration, the function returns None.
    fn fpt(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        max_duration: impl Into<f64>,
    ) -> XResult<Option<f64>> {
        let fpt = FirstPassageTime::new(self, domain)?;
        fpt.simulate_p(max_duration)
    }

    /// Get the occupation time of the point process
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain which the occupation time is interested in.
    /// * `duration` - The duration of the simulation.
    fn occupation_time(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        duration: impl Into<f64>,
    ) -> XResult<f64> {
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

/// Continuous trajectory
pub struct ContinuousTrajectory<SP: ContinuousProcess> {
    /// The continuous process
    sp: SP,
    /// The duration of the simulation
    duration: f64,
}

impl<SP: ContinuousProcess> ContinuousTrajectory<SP> {
    /// Create a new `ContinuousTrajectory` with given `ContinuousProcess` and duration.
    ///
    /// # Arguments
    ///
    /// * `sp` - The continuous process.
    /// * `duration` - The duration of the simulation.
    pub fn new(sp: SP, duration: impl Into<f64>) -> XResult<Self> {
        let duration = duration.into();
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {}",
                duration
            ))
            .into());
        }
        Ok(Self { sp, duration })
    }

    /// Get the continuous process
    pub fn process(&self) -> &SP {
        &self.sp
    }

    /// Get the duration of the trajectory
    pub fn duration(&self) -> f64 {
        self.duration
    }

    /// Simulate method
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the simulation.
    pub fn simulate(&self, time_step: f64) -> XResult<Pair> {
        self.sp.simulate(self.duration, time_step)
    }
}

/// Discrete trajectory
pub struct DiscreteTrajectory<SP: DiscreteProcess> {
    /// The discrete process
    sp: SP,
    /// The number of steps
    num_step: usize,
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
    pub fn process(&self) -> &SP {
        &self.sp
    }

    /// Get the number of steps of the trajectory
    pub fn num_step(&self) -> usize {
        self.num_step
    }

    /// Simulate method
    pub fn simulate(&self) -> XResult<DiscretePair> {
        self.sp.simulate(self.num_step)
    }
}

/// Point process trajectory
pub struct PointTrajectory<SP: PointProcess> {
    /// The point process
    sp: SP,
    /// The duration of the trajectory
    duration: Option<f64>,
    /// The number of steps of the trajectory
    num_step: Option<usize>,
}

impl<SP: PointProcess> PointTrajectory<SP> {
    /// Get the point process
    pub fn process(&self) -> &SP {
        &self.sp
    }

    /// Get the duration of the trajectory
    pub fn duration(&self) -> Option<f64> {
        self.duration
    }

    /// Get the number of steps of the trajectory
    pub fn num_step(&self) -> Option<usize> {
        self.num_step
    }

    /// Create a `PointTrajectory` with duration.
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the trajectory
    pub fn with_duration(sp: SP, duration: impl Into<f64>) -> XResult<Self> {
        let duration = duration.into();
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {}",
                duration
            ))
            .into());
        }
        Ok(Self {
            sp,
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
            sp,
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

/// Continuous trajectory trait
pub trait ContinuousTrajectoryTrait: ContinuousProcess {
    /// Create a `ContinuousTrajectory` with given duration
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the trajectory
    fn duration(&self, duration: impl Into<f64>) -> XResult<ContinuousTrajectory<Self>> {
        let traj = ContinuousTrajectory::new(self.clone(), duration)?;
        Ok(traj)
    }
}

impl<SP: ContinuousProcess> ContinuousTrajectoryTrait for SP {}

/// Discrete trajectory trait
pub trait DiscreteTrajectoryTrait: DiscreteProcess {
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

impl<SP: DiscreteProcess> DiscreteTrajectoryTrait for SP {}

/// Point trajectory trait
pub trait PointTrajectoryTrait: PointProcess {
    /// Create a `PointTrajectory` with given duration
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the trajectory
    fn duration(&self, duration: impl Into<f64>) -> XResult<PointTrajectory<Self>> {
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
    fn raw_moment(&self, order: i32, particles: usize, time_step: f64) -> XResult<f64>;

    /// Get the central moment of the simulation
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    fn central_moment(&self, order: i32, particles: usize, time_step: f64) -> XResult<f64>;
}

impl<SP: ContinuousProcess> Moment for ContinuousTrajectory<SP> {
    fn raw_moment(&self, order: i32, particles: usize, time_step: f64) -> XResult<f64> {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {}",
                particles
            ))
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
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {}",
                particles
            ))
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
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {}",
                particles
            ))
            .into());
        }

        if order == 0 {
            return Ok(0.0);
        }

        let sp = self.sp.clone();

        if self.duration.is_none() {
            return Err(SimulationError::InvalidParameters(
                "The `duration` must be provided".to_string(),
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

/// Inverse process of a continuous process
#[derive(Clone)]
pub struct InverseProcess<T: ContinuousProcess> {
    /// The process
    process: T,
}

impl<T: ContinuousProcess> InverseProcess<T> {
    /// Create a new inverse process with given process
    ///
    /// # Arguments
    ///
    /// * `process` - The continuous process
    pub fn new(process: &T) -> Self {
        Self {
            process: process.clone(),
        }
    }

    /// Get the process
    pub fn process(&self) -> &T {
        &self.process
    }
}

impl<T: ContinuousProcess> ContinuousProcess for InverseProcess<T> {
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        let mut mut_duration = duration.into();
        let duration = mut_duration;
        let sp = self.process.clone();
        let (t, s) = loop {
            let (t, s) = sp.simulate(mut_duration, time_step)?;
            let last = match s.last() {
                Some(x) => *x,
                None => return Err(SimulationError::Unknown.into()),
            };
            if last >= duration {
                break (t, s);
            }
            mut_duration *= 2.0;
        };

        let num_steps = (duration / time_step).ceil() as usize;
        let inv_times: Vec<f64> = (0..=num_steps)
            .map(|i| {
                if i == num_steps {
                    duration
                } else {
                    i as f64 * time_step
                }
            })
            .collect();

        let mut inv_path = Vec::with_capacity(inv_times.len());

        for &target_time in &inv_times {
            let pos = match s.binary_search_by(|&x| x.partial_cmp(&target_time).unwrap()) {
                Ok(idx) => idx,
                Err(idx) => {
                    if idx >= s.len() {
                        s.len() - 1
                    } else {
                        idx
                    }
                }
            };

            inv_path.push(t[pos]);
        }

        Ok((inv_times, inv_path))
    }
}

/// The inverse process trait
pub trait Inverse: ContinuousProcess {
    /// Create a new `InverseProcess`
    fn inverse(&self) -> InverseProcess<Self> {
        InverseProcess::new(self)
    }
}

/// TAMSD (time-averaged mean-squared displacement)
#[derive(Debug, Clone)]
pub struct TAMSD<SP: Clone> {
    /// The continuous process
    process: SP,
    /// The duration
    duration: f64,
    /// The slag length
    delta: f64,
}

impl<SP: Clone> TAMSD<SP> {
    /// Create a new TAMSD
    ///
    /// # Arguments
    ///
    /// * `process` - The continuous process to calculate the TAMSD of.
    /// * `duration` - The duration of the simulation.
    /// * `delta` - The slag length.
    pub fn new(process: &SP, duration: impl Into<f64>, delta: impl Into<f64>) -> XResult<Self> {
        let duration = duration.into();
        let delta = delta.into();
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {}",
                duration
            ))
            .into());
        }
        if delta <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `delta` must be positive, got {}",
                delta
            ))
            .into());
        }

        if duration < delta {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be greater than `delta`, got duration {} and delta {}",
                duration, delta
            ))
            .into());
        }
        Ok(Self {
            process: process.clone(),
            duration,
            delta,
        })
    }

    /// Get the process
    pub fn process(&self) -> &SP {
        &self.process
    }

    /// Get the duration
    pub fn duration(&self) -> f64 {
        self.duration
    }

    /// Get the slag length
    pub fn delta(&self) -> f64 {
        self.delta
    }
}

impl<SP: ContinuousProcess> TAMSD<SP> {
    /// Simulate the TAMSD
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    pub fn simulate(&self, time_step: f64, quad_order: usize) -> XResult<f64> {
        let legendre_quad = GaussLegendre::new(quad_order)?;
        let nodes_weights_pairs = legendre_quad.into_node_weight_pairs();
        let duration = self.duration;
        let slag = self.delta;
        let nodes_weights = nodes_weights_transform(0.0, duration - slag, &nodes_weights_pairs);
        let sp = self.process.clone();
        let result = nodes_weights
            .into_par_iter()
            .map(|(node, weight)| -> XResult<f64> {
                let slag_length = (slag / time_step).ceil() as usize;
                let (_, x) = sp.simulate(node + slag, time_step)?;
                let len = x.len();
                let end_position = x.last();
                let slag_position = x.get(len - slag_length - 1);
                if end_position.is_none() || slag_position.is_none() {
                    return Err(SimulationError::Unknown.into());
                }
                let end_position = *end_position.unwrap();
                let slag_position = *slag_position.unwrap();

                Ok((end_position - slag_position).powi(2) * weight)
            })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / (duration - slag);
        Ok(result)
    }

    /// Get the ensemble average of the TAMSD
    ///
    /// # Arguments
    ///
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    pub fn mean(&self, particles: usize, time_step: f64, quad_order: usize) -> XResult<f64> {
        Ok((0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> { self.simulate(time_step, quad_order) })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / particles as f64)
    }

    /// Get the variance of the TAMSD
    ///
    /// # Arguments
    ///
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    pub fn variance(&self, particles: usize, time_step: f64, quad_order: usize) -> XResult<f64> {
        let mean = self.mean(particles, time_step, quad_order)?;
        Ok((0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let value = self.simulate(time_step, quad_order)?;
                Ok((value - mean).powi(2))
            })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / particles as f64)
    }
}

impl<SP: PointProcess> TAMSD<SP> {
    /// Simulate the TAMSD
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    pub fn simulate_p(&self, time_step: f64, quad_order: usize) -> XResult<f64> {
        let legendre_quad = GaussLegendre::new(quad_order)?;
        let nodes_weights_pairs = legendre_quad.into_node_weight_pairs();
        let duration = self.duration;
        let slag = self.delta;
        let nodes_weights = nodes_weights_transform(0.0, duration - slag, &nodes_weights_pairs);
        let sp = self.process.clone();
        let result = nodes_weights
            .into_par_iter()
            .map(|(node, weight)| -> XResult<f64> {
                let slag_length = (slag / time_step).ceil() as usize;
                let (t, x) = sp.simulate_with_duration(node + slag)?;
                let (_, x) = flatten_interpolate(&t, &x, time_step)?;
                let len = x.len();
                let end_position = x.last();
                let slag_position = x.get(len - slag_length - 1);
                if end_position.is_none() || slag_position.is_none() {
                    return Err(SimulationError::Unknown.into());
                }
                let end_position = *end_position.unwrap();
                let slag_position = *slag_position.unwrap();

                Ok((end_position - slag_position).powi(2) * weight)
            })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / (duration - slag);
        Ok(result)
    }

    /// Get the ensemble average of the TAMSD
    ///
    /// # Arguments
    ///
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    pub fn mean_p(&self, particles: usize, time_step: f64, quad_order: usize) -> XResult<f64> {
        Ok((0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> { self.simulate_p(time_step, quad_order) })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / particles as f64)
    }

    /// Get the variance of the TAMSD
    ///
    /// # Arguments
    ///
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    pub fn variance_p(&self, particles: usize, time_step: f64, quad_order: usize) -> XResult<f64> {
        let mean = self.mean_p(particles, time_step, quad_order)?;
        Ok((0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let value = self.simulate_p(time_step, quad_order)?;
                Ok((value - mean).powi(2))
            })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / particles as f64)
    }
}

/// Transform the nodes and weights
///
/// # Arguments
///
/// * `a` - The lower bound of the interval.
/// * `b` - The upper bound of the interval.
/// * `pairs` - The nodes and weights pairs of the unit interval.
fn nodes_weights_transform(
    a: impl Into<f64>,
    b: impl Into<f64>,
    pairs: &[(f64, f64)],
) -> Vec<(f64, f64)> {
    let a: f64 = a.into();
    let b: f64 = b.into();
    pairs
        .iter()
        .map(|(node, weight)| {
            let new_weight = weight * (b - a) / 2.0;
            let new_node = (b - a) * node / 2.0 + (b + a) / 2.0;
            (new_node, new_weight)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::continuous::Bm;

    #[test]
    fn test_quad() {
        let quad = GaussLegendre::new(10).unwrap();
        let nodes_weights_pairs = quad.into_node_weight_pairs();
        let nodes_weights = nodes_weights_transform(0.0, 100.0, &nodes_weights_pairs);
        println!("{:?}", nodes_weights);
    }

    #[test]
    #[ignore]
    fn test_continuous_trajectory() {
        let sp = Bm::default();
        let tamsd = TAMSD::new(&sp, 100.0, 1.0).unwrap();
        let value = tamsd.simulate(0.1, 10).unwrap();
        println!("{:?}", value);
        let eatamsd = tamsd.mean(10000, 0.1, 10).unwrap();
        println!("{:?}", eatamsd);
    }
}
