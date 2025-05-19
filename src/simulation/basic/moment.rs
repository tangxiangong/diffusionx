use super::{
    ContinuousProcess, ContinuousTrajectory, DiscreteProcess, DiscreteTrajectory, PointProcess,
    PointTrajectory,
};
use crate::{SimulationError, XResult};
use rayon::prelude::*;

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

impl<'a, SP: ContinuousProcess> Moment for ContinuousTrajectory<'a, SP> {
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

        let duration = self.duration;

        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let (_, x) = self.sp.simulate(duration, time_step)?;
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
        let duration = self.duration;
        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let (_, x) = self.sp.simulate(duration, time_step)?;
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

impl<'a, SP: DiscreteProcess> Moment for DiscreteTrajectory<'a, SP> {
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

        let num_step = self.num_step;

        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let (_, x) = self.sp.simulate(num_step)?;
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
        let num_step = self.num_step;
        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let (_, x) = self.sp.simulate(num_step)?;
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

impl<'a, SP: PointProcess> Moment for PointTrajectory<'a, SP> {
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
                let (_, x) = self.sp.simulate_with_duration(duration)?;
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
        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let (_, x) = self.sp.simulate_with_duration(duration)?;
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
