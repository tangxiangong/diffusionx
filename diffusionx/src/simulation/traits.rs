use crate::{SimulationError, XResult};
use rayon::prelude::*;

pub type Pair = (Vec<f64>, Vec<f64>);

/// Simulation trait
///
/// This trait represents a simulation.
pub trait Simulation: Clone {
    /// Get the duration of the simulation
    fn get_duration(&self) -> f64;
    /// Set the duration of the simulation
    fn mut_duration(&mut self, duration: f64);
    /// Simulate the simulation
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the simulation.
    fn simulate(&self, time_step: f64) -> XResult<Pair>;

    /// Simulate the simulation with check
    fn simulate_check(&self, time_step: f64) -> XResult<Pair>;
}

/// CheckedParams trait
///
/// This trait represents a simulation that has checked parameters.
pub trait CheckedParams: Simulation {
    /// Check the parameters of the simulation
    fn check_params(&self, time_step: f64) -> XResult<()>;
}

/// Moment trait
///
/// This trait represents a simulation that has moments.
pub trait Moment: Simulation + CheckedParams + Send + Sync {
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
    fn raw_moment(&self, time_step: f64, order: i32, particles: usize) -> XResult<f64> {
        self.check_params(time_step)?;
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

        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let (_, x) = self.simulate(time_step)?;
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
    fn central_moment(&self, time_step: f64, order: i32, particles: usize) -> XResult<f64> {
        let mean = self.raw_moment(time_step, 1, particles)?;
        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let (_, x) = self.simulate(time_step)?;
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

/// Functional trait
///
/// This trait represents a simulation that has functional properties.
pub trait Functional: Simulation + CheckedParams {
    /// Get the first passage time of the simulation
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the simulation.
    /// * `domain` - The domain of the simulation.
    /// * `max_duration` - The maximum duration of the simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the first passage time of the simulation.
    fn fpt_check(
        &self,
        time_step: f64,
        domain: (impl Into<f64>, impl Into<f64>),
        max_duration: impl Into<f64>,
    ) -> XResult<Option<f64>> {
        let (a, b) = domain;
        let a = a.into();
        let b = b.into();
        let max_duration = max_duration.into();
        if a >= b {
            return Err(SimulationError::InvalidParameters(
                "domain must be a valid interval".to_string(),
            )
            .into());
        }
        if max_duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(
                "max_duration must be positive".to_string(),
            )
            .into());
        }
        self.check_params(time_step)?;
        self.fpt(time_step, (a, b), max_duration)
    }

    /// Get the first passage time of the simulation
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the simulation.
    /// * `domain` - The domain of the simulation.
    /// * `max_duration` - The maximum duration of the simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the first passage time of the simulation.
    fn fpt(
        &self,
        time_step: f64,
        domain: (impl Into<f64>, impl Into<f64>),
        max_duration: impl Into<f64>,
    ) -> XResult<Option<f64>> {
        let (a, b) = domain;
        let a = a.into();
        let b = b.into();
        let max_duration = max_duration.into();
        let mut duration = self.get_duration();
        let mut tmp = self.clone();
        loop {
            let (t, x) = tmp.simulate(time_step)?;
            if let Some(index) = x.iter().position(|&x| x <= a || x >= b) {
                return Ok(Some(t[index]));
            }
            duration *= 2.0;
            if duration > max_duration {
                tmp.mut_duration(max_duration);
                let (t, x) = tmp.simulate(time_step)?;
                if let Some(index) = x.iter().position(|&x| x <= a || x >= b) {
                    return Ok(Some(t[index]));
                } else {
                    return Ok(None);
                }
            }
            tmp.mut_duration(duration);
        }
    }
}
