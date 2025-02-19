use crate::{SimulationError, XResult, simulation::Simulation};
use rayon::prelude::*;

/// Functional for first passage time
/// 
/// # Fields
///
/// * `sp` - The simulation object.
/// * `domain` - The domain of the simulation.
pub struct FirstPassageTime<SP: Simulation> {
    sp: SP,
    domain: (f64, f64),
}

impl<SP: Simulation> FirstPassageTime<SP> {
    pub fn new(sp: &SP, domain: (impl Into<f64>, impl Into<f64>)) -> XResult<Self> {
        let domain = (domain.0.into(), domain.1.into());
        if domain.0 >= domain.1 {
            return Err(SimulationError::InvalidParameters(
                "domain must be a valid interval".to_string(),
            )
            .into());
        }
        Ok(Self {
            sp: sp.clone(),
            domain,
        })
    }

    /// Simulate the first passage time
    /// 
    /// # Arguments
    /// 
    /// * `max_duration` - The maximum duration of the simulation.
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Returns
    /// 
    /// `Option<f64>`
    /// * None if the first passage time is not found within the maximum duration.
    /// * A f64 representing the first passage time of the simulation.
    ///
    /// # Example
    /// 
    /// ```rust
    /// let fpt = FirstPassageTime::new(&sp, (0.0, 1.0)).unwrap();
    /// let fpt_result = fpt.simulate(1000.0, 0.1).unwrap();
    /// ```
    pub fn simulate(&self, max_duration: impl Into<f64>, time_step: f64) -> XResult<Option<f64>> {
        if time_step <= 0.0 {
            return Err(SimulationError::InvalidParameters(
                "time_step must be positive".to_string(),
            )
            .into());
        }
        let (a, b) = self.domain;
        let sp = self.sp.clone();
        let max_duration = max_duration.into();
        let mut duration = 10.0;
        loop {
            let (t, x) = sp.simulate(duration, time_step)?;
            if let Some(index) = x.iter().position(|&x| x <= a || x >= b) {
                return Ok(Some(t[index]));
            }
            duration *= 2.0;
            if duration > max_duration {
                duration = max_duration;
                let (t, x) = sp.simulate(duration, time_step)?;
                if let Some(index) = x.iter().position(|&x| x <= a || x >= b) {
                    return Ok(Some(t[index]));
                } else {
                    return Ok(None);
                }
            }
        }
    }
    
    /// Get the raw moment of the first passage time
    /// 
    /// # Arguments
    /// 
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `max_duration` - The maximum duration of the simulation.
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Returns
    /// 
    /// `Option<f64>`
    /// * None if the raw moment is not found.
    /// * A f64 representing the raw moment of the simulation.
    ///
    /// # Example
    /// 
    /// ```rust
    /// let fpt = FirstPassageTime::new(&sp, (0.0, 1.0)).unwrap();
    /// let raw_moment = fpt.raw_moment(1, 1000, 1000.0, 0.1).unwrap();
    /// ```
    pub fn raw_moment(
        &self,
        order: i32,
        particles: usize,
        max_duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<Option<f64>> {
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
            return Ok(Some(0.0));
        }
        let max_duration = max_duration.into();
        if max_duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(
                "max_duration must be positive".to_string(),
            )
            .into());
        }

        // 使用元组来同时跟踪总和和有效样本数
        let (sum, valid_count) = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<Option<f64>> {
                let fpt = self.simulate(max_duration, time_step)?;
                match fpt {
                    Some(t) => Ok(Some(t.powi(order))),
                    None => Ok(None),
                }
            })
            .try_fold(
                || (0.0, 0usize),
                |acc, res| -> XResult<(f64, usize)> {
                    match res? {
                        Some(v) => Ok((acc.0 + v, acc.1 + 1)),
                        None => Ok(acc),
                    }
                },
            )
            .try_reduce(|| (0.0, 0usize), |a, b| Ok((a.0 + b.0, a.1 + b.1)))?;

        if valid_count == 0 {
            Ok(None)
        } else {
            Ok(Some(sum / valid_count as f64))
        }
    }

    /// Get the central moment of the first passage time
    /// 
    /// # Arguments
    /// 
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `max_duration` - The maximum duration of the simulation.
    /// * `time_step` - The time step of the simulation.    
    ///
    /// # Returns
    /// 
    /// `Option<f64>`
    /// * None if the central moment is not found.
    /// * A f64 representing the central moment of the simulation.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// let fpt = FirstPassageTime::new(&sp, (0.0, 1.0)).unwrap();
    /// let central_moment = fpt.central_moment(1, 1000, 1000.0, 0.1).unwrap();
    /// ```
    pub fn central_moment(
        &self,
        order: i32,
        particles: usize,
        max_duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<Option<f64>> {
        let max_duration = max_duration.into();
        let mean = self.raw_moment(order, particles, max_duration, time_step)?;
        if mean.is_none() {
            return Ok(None);
        }
        let mean = mean.unwrap();
        let (sum, valid_count) = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<Option<f64>> {
                let fpt = self.simulate(max_duration, time_step)?;
                match fpt {
                    Some(t) => Ok(Some((t - mean).powi(order))),
                    None => Ok(None),
                }
            })
            .try_fold(
                || (0.0, 0usize),
                |acc, res| -> XResult<(f64, usize)> {
                    match res? {
                        Some(v) => Ok((acc.0 + v, acc.1 + 1)),
                        None => Ok(acc),
                    }
                },
            )
            .try_reduce(|| (0.0, 0usize), |a, b| Ok((a.0 + b.0, a.1 + b.1)))?;

        if valid_count == 0 {
            Ok(None)
        } else {
            Ok(Some(sum / valid_count as f64))
        }
    }
}
