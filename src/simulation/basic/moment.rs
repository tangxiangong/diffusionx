use super::{
    ContinuousProcess, ContinuousTrajectory, DiscreteProcess, DiscreteTrajectory, PointProcess,
    PointTrajectory,
};
use crate::{SimulationError, XResult};
use num_traits::Float;
use rayon::prelude::*;
use std::fmt::Debug;

/// Moment trait
pub trait Moment<T: Float + Send + Sync = f64> {
    /// Get the raw moment of the simulation
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    fn raw_moment(&self, order: i32, particles: usize, time_step: T) -> XResult<T>;

    /// Get the central moment of the simulation
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    fn central_moment(&self, order: i32, particles: usize, time_step: T) -> XResult<T>;
}

impl<T: Float + Send + Sync + Debug + std::iter::Sum, SP: ContinuousProcess<T> + Clone> Moment<T>
    for ContinuousTrajectory<SP, T>
{
    fn raw_moment(&self, order: i32, particles: usize, time_step: T) -> XResult<T> {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {particles}"
            ))
            .into());
        }

        if order == 0 {
            return Ok(T::one());
        }
        if time_step <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `time_step` must be positive, got `{time_step:?}`"
            ))
            .into());
        }

        let duration = self.duration;

        let values = (0..particles)
            .into_par_iter()
            .map(|_| {
                let end = match self.sp.end(duration, time_step) {
                    Ok(end) => end,
                    Err(e) => panic!("{}", e),
                };
                if order == 1 { end } else { end.powi(order) }
            })
            .sum::<T>();

        let result = values / (T::from(particles).unwrap());
        Ok(result)
    }

    fn central_moment(&self, order: i32, particles: usize, time_step: T) -> XResult<T> {
        if order == 0 {
            return Ok(T::one());
        }
        if time_step <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `time_step` must be positive, got `{time_step:?}`"
            ))
            .into());
        }
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {particles}"
            ))
            .into());
        }
        let mean = self.raw_moment(1, particles, time_step)?;
        let duration = self.duration;
        let values = (0..particles)
            .into_par_iter()
            .map(|_| {
                let end_position = match self.sp.end(duration, time_step) {
                    Ok(end_position) => end_position,
                    Err(e) => panic!("{}", e),
                };
                if order == 1 {
                    end_position - mean
                } else {
                    (end_position - mean).powi(order)
                }
            })
            .sum::<T>();

        let result = values / (T::from(particles).unwrap());
        Ok(result)
    }
}

impl<SP: DiscreteProcess + Clone> Moment for DiscreteTrajectory<SP> {
    fn raw_moment(&self, order: i32, particles: usize, _: f64) -> XResult<f64> {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {particles}"
            ))
            .into());
        }

        if order == 0 {
            return Ok(1.0);
        }

        let num_step = self.num_step;

        let values = (0..particles)
            .into_par_iter()
            .map(|_| {
                let end_position = match self.sp.end(num_step) {
                    Ok(end_position) => end_position,
                    Err(e) => panic!("{}", e),
                };
                if order == 1 {
                    end_position
                } else {
                    end_position.powi(order)
                }
            })
            .sum::<f64>();

        let result = values / (particles as f64);
        Ok(result)
    }

    fn central_moment(&self, order: i32, particles: usize, _: f64) -> XResult<f64> {
        if order == 0 {
            return Ok(1.0);
        }

        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {particles}"
            ))
            .into());
        }

        let mean = self.raw_moment(1, particles, 0.01)?;
        let num_step = self.num_step;
        let values = (0..particles)
            .into_par_iter()
            .map(|_| {
                let end_position = match self.sp.end(num_step) {
                    Ok(end_position) => end_position,
                    Err(e) => panic!("{}", e),
                };
                if order == 1 {
                    end_position - mean
                } else {
                    (end_position - mean).powi(order)
                }
            })
            .sum::<f64>();

        let result = values / (particles as f64);
        Ok(result)
    }
}

impl<SP: PointProcess> Moment for PointTrajectory<SP> {
    fn raw_moment(&self, order: i32, particles: usize, _: f64) -> XResult<f64> {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {particles}"
            ))
            .into());
        }

        if order == 0 {
            return Ok(1.0);
        }

        if self.duration.is_none() {
            return Err(SimulationError::InvalidParameters(
                "The `duration` must be provided".to_string(),
            )
            .into());
        }
        let duration = self.duration.unwrap();

        let values = (0..particles)
            .into_par_iter()
            .map(|_| {
                let end_position = match self.sp.end(duration) {
                    Ok(end_position) => end_position,
                    Err(e) => panic!("{}", e),
                };
                if order == 1 {
                    end_position
                } else {
                    end_position.powi(order)
                }
            })
            .sum::<f64>();

        let result = values / (particles as f64);
        Ok(result)
    }

    fn central_moment(&self, order: i32, particles: usize, _time_step: f64) -> XResult<f64> {
        if order == 0 {
            return Ok(1.0);
        }

        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {particles}"
            ))
            .into());
        }

        let mean = self.raw_moment(1, particles, _time_step)?;
        let duration = self.duration.unwrap();
        let values = (0..particles)
            .into_par_iter()
            .map(|_| {
                let end_position = match self.sp.end(duration) {
                    Ok(end_position) => end_position,
                    Err(e) => panic!("{}", e),
                };
                if order == 1 {
                    end_position - mean
                } else {
                    (end_position - mean).powi(order)
                }
            })
            .sum::<f64>();

        let result = values / (particles as f64);
        Ok(result)
    }
}
