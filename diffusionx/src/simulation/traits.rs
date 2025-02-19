use crate::{SimulationError, XResult};
use rayon::prelude::*;

pub type Pair = (Vec<f64>, Vec<f64>);

/// Stochastic trait
pub trait Stochastic: Clone + Send + Sync {}

/// Simulation trait
pub trait Simulation: Stochastic {
    /// Simulate the simulation
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the simulation.
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair>;
}

/// Stochastic trajectory
/// 
/// # Fields
///
/// * `sp` - The simulation object.
/// * `duration` - The duration of the simulation.
pub struct StochasticTrajectory<SP: Simulation> {
    pub(crate) sp: SP,
    pub(crate) duration: f64,
}

impl<SP: Simulation> StochasticTrajectory<SP> {
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

/// Trajectory trait
pub trait Trajectory: Simulation {
    fn duration(&self, duration: impl Into<f64>) -> XResult<StochasticTrajectory<Self>> {
        let traj = StochasticTrajectory::new(self.clone(), duration)?;
        Ok(traj)
    }
}

impl<SP: Simulation> Trajectory for SP {}

/// Moment trait
pub trait Moment {
    /// Get the raw moment of the simulation
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the simulation.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    ///
    /// # Returns
    ///
    /// A f64 representing the raw moment of the simulation.
    fn raw_moment(&self, order: i32, particles: usize, time_step: f64) -> XResult<f64>;

    /// Get the central moment of the simulation
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the simulation.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    ///
    /// # Returns
    ///
    /// A f64 representing the central moment of the simulation.
    fn central_moment(&self, order: i32, particles: usize, time_step: f64) -> XResult<f64>;
}

impl<SP: Simulation> Moment for StochasticTrajectory<SP> {
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
