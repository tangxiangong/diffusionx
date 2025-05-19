//! Functional distribution of stochastic processes
//!
//! - First passage time [FirstPassageTime]
//! - Occupation time [OccupationTime]
//!

use crate::{
    SimulationError, XResult,
    simulation::prelude::{ContinuousProcess, PointProcess},
};
use rayon::prelude::*;

/// First passage time
#[derive(Debug, Clone)]
pub struct FirstPassageTime<'a, SP> {
    /// The stochastic process
    sp: &'a SP,
    /// The domain that the first passage time is interested in
    domain: (f64, f64),
}

impl<'a, SP: Send + Sync> FirstPassageTime<'a, SP> {
    /// Create a new first passage time
    ///
    /// # Arguments
    ///
    /// * `sp` - The stochastic process.
    /// * `domain` - The domain that the first passage time is interested in.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::functional::FirstPassageTime;
    /// use diffusionx::simulation::continuous::Bm;
    ///
    /// let sp = Bm::default();
    /// let fpt = FirstPassageTime::new(&sp, (0.0, 1.0)).unwrap();
    /// ```
    pub fn new(sp: &'a SP, domain: (impl Into<f64>, impl Into<f64>)) -> XResult<Self> {
        let domain = (domain.0.into(), domain.1.into());
        if domain.0 >= domain.1 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `domain` must be a valid interval, i.e., `domain.0 < domain.1`, got `{:?}`",
                domain
            ))
            .into());
        }
        Ok(Self { sp, domain })
    }
}

impl<'a, SP: ContinuousProcess> FirstPassageTime<'a, SP> {
    /// Simulate the first passage time
    ///
    /// # Arguments
    ///
    /// * `max_duration` - The maximum duration of the simulation. If the first passage time is not achieved within the maximum duration, the function will return `None`.
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::functional::FirstPassageTime;
    /// use diffusionx::simulation::continuous::Bm;
    ///
    /// let sp = Bm::default();
    /// let fpt = FirstPassageTime::new(&sp, (0.0, 1.0)).unwrap();
    /// let result = fpt.simulate(1000.0, 0.1).unwrap();
    /// ```
    pub fn simulate(&self, max_duration: impl Into<f64>, time_step: f64) -> XResult<Option<f64>> {
        if time_step <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `time_step` must be positive, got `{}`",
                time_step
            ))
            .into());
        }
        let (a, b) = self.domain;
        let max_duration = max_duration.into();
        let mut duration = 10.0;
        loop {
            let (t, x) = self.sp.simulate(duration, time_step)?;
            if let Some(index) = x.iter().position(|&x| x <= a || x >= b) {
                return Ok(Some(t[index]));
            }
            duration *= 2.0;
            if duration > max_duration {
                duration = max_duration;
                let (t, x) = self.sp.simulate(duration, time_step)?;
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
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::functional::FirstPassageTime;
    /// use diffusionx::simulation::continuous::Bm;
    ///
    /// let sp = Bm::default();
    /// let fpt = FirstPassageTime::new(&sp, (0.0, 1.0)).unwrap();
    /// let result = fpt.raw_moment(1, 1000, 1000.0, 0.1).unwrap();
    /// ```
    pub fn raw_moment(
        &self,
        order: i32,
        particles: u64,
        max_duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<Option<f64>> {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got `{}`",
                particles
            ))
            .into());
        }
        let max_duration = max_duration.into();
        if max_duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `max_duration` must be positive, got `{}`",
                max_duration
            ))
            .into());
        }

        // Use tuple to track the sum and the number of valid samples
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
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::functional::FirstPassageTime;
    /// use diffusionx::simulation::continuous::Bm;
    ///
    /// let sp = Bm::default();
    /// let fpt = FirstPassageTime::new(&sp, (0.0, 1.0)).unwrap();
    /// let result = fpt.central_moment(1, 1000, 1000.0, 0.1).unwrap();
    /// ```
    pub fn central_moment(
        &self,
        order: i32,
        particles: u64,
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

/// Occupation time
#[derive(Debug, Clone)]
pub struct OccupationTime<'a, SP> {
    /// The stochastic process
    sp: &'a SP,
    /// The domain that the occupation time is interested in
    domain: (f64, f64),
    /// The duration of the simulation
    duration: f64,
}

impl<'a, SP: Send + Sync> OccupationTime<'a, SP> {
    /// Create a new occupation time
    ///
    /// # Arguments
    ///
    /// * `sp` - The stochastic process.
    /// * `domain` - The domain that the occupation time is interested in.
    /// * `duration` - The duration of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::functional::OccupationTime;
    /// use diffusionx::simulation::continuous::Bm;
    ///
    /// let sp = Bm::default();
    /// let ot = OccupationTime::new(&sp, (0.0, 1.0), 1000.0).unwrap();
    /// ```
    pub fn new(
        sp: &'a SP,
        domain: (impl Into<f64>, impl Into<f64>),
        duration: impl Into<f64>,
    ) -> XResult<Self> {
        let domain = (domain.0.into(), domain.1.into());
        let duration = duration.into();
        if domain.0 >= domain.1 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `domain` must be a valid interval, i.e., `domain.0 < domain.1`, got `{:?}`",
                domain
            ))
            .into());
        }
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got `{}`",
                duration
            ))
            .into());
        }
        Ok(Self {
            sp,
            domain,
            duration,
        })
    }
}

impl<'a, SP: ContinuousProcess> OccupationTime<'a, SP> {
    /// Simulate the occupation time
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::functional::OccupationTime;
    /// use diffusionx::simulation::continuous::Bm;
    ///
    /// let sp = Bm::default();
    /// let ot = OccupationTime::new(&sp, (0.0, 1.0), 1000.0).unwrap();
    /// let result = ot.simulate(0.1).unwrap();
    /// ```
    pub fn simulate(&self, time_step: f64) -> XResult<f64> {
        let (t, x) = self.sp.simulate(self.duration, time_step)?;
        let (a, b) = self.domain;

        let occupation_time = x
            .windows(2)
            .zip(t.windows(2))
            .map(|(x_pair, t_pair)| {
                let in_domain = (a..=b).contains(&x_pair[0]) && (a..=b).contains(&x_pair[1]);
                if in_domain {
                    t_pair[1] - t_pair[0]
                } else {
                    0.0
                }
            })
            .sum();

        Ok(occupation_time)
    }

    /// Get the raw moment of the occupation time
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::functional::OccupationTime;
    /// use diffusionx::simulation::continuous::Bm;
    ///
    /// let sp = Bm::default();
    /// let ot = OccupationTime::new(&sp, (0.0, 1.0), 1000.0).unwrap();
    /// let result = ot.raw_moment(1, 1000, 0.1).unwrap();
    /// ```
    pub fn raw_moment(&self, order: i32, particles: usize, time_step: f64) -> XResult<f64> {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got `{}`",
                particles
            ))
            .into());
        }
        if order < 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `order` must be non-negative, got `{}`",
                order
            ))
            .into());
        }
        if order == 0 {
            return Ok(0.0);
        }

        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let occupation_time = self.simulate(time_step)?;
                Ok(occupation_time.powi(order))
            })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / particles as f64;
        Ok(result)
    }

    /// Get the central moment of the occupation time
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::functional::OccupationTime;
    /// use diffusionx::simulation::continuous::Bm;
    ///
    /// let sp = Bm::default();
    /// let ot = OccupationTime::new(&sp, (0.0, 1.0), 1000.0).unwrap();
    /// let result = ot.central_moment(1, 1000, 0.1).unwrap();
    /// ```
    pub fn central_moment(&self, order: i32, particles: usize, time_step: f64) -> XResult<f64> {
        let mean = self.raw_moment(order, particles, time_step)?;
        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let occupation_time = self.simulate(time_step)?;
                Ok((occupation_time - mean).powi(order))
            })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / particles as f64;
        Ok(result)
    }
}

impl<'a, SP: PointProcess> FirstPassageTime<'a, SP> {
    /// Simulate the first passage time
    ///
    /// # Arguments
    ///
    /// * `max_duration` - The maximum duration of the simulation.
    pub fn simulate_p(&self, max_duration: impl Into<f64>) -> XResult<Option<f64>> {
        let (a, b) = self.domain;
        let max_duration = max_duration.into();
        let mut duration = 10.0;
        loop {
            let (t, x) = self.sp.simulate_with_duration(duration)?;
            if let Some(index) = x.iter().position(|&x| x <= a || x >= b) {
                return Ok(Some(t[index]));
            }
            duration *= 2.0;
            if duration > max_duration {
                duration = max_duration;
                let (t, x) = self.sp.simulate_with_duration(duration)?;
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
    pub fn raw_moment_p(
        &self,
        order: i32,
        particles: usize,
        max_duration: impl Into<f64>,
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
                let fpt = self.simulate_p(max_duration)?;
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
    pub fn central_moment_p(
        &self,
        order: i32,
        particles: usize,
        max_duration: impl Into<f64>,
    ) -> XResult<Option<f64>> {
        let max_duration = max_duration.into();
        let mean = self.raw_moment_p(order, particles, max_duration)?;
        if mean.is_none() {
            return Ok(None);
        }
        let mean = mean.unwrap();
        let (sum, valid_count) = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<Option<f64>> {
                let fpt = self.simulate_p(max_duration)?;
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

impl<'a, SP: PointProcess> OccupationTime<'a, SP> {
    pub fn simulate_p(&self) -> XResult<f64> {
        let (t, x) = self.sp.simulate_with_duration(self.duration)?;
        let (a, b) = self.domain;

        let occupation_time = x
            .windows(2)
            .zip(t.windows(2))
            .map(|(x_pair, t_pair)| {
                let in_domain = (a..=b).contains(&x_pair[0]) && (a..=b).contains(&x_pair[1]);
                if in_domain {
                    t_pair[1] - t_pair[0]
                } else {
                    0.0
                }
            })
            .sum();

        Ok(occupation_time)
    }

    /// Get the raw moment of the occupation time
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    pub fn raw_moment_p(&self, order: i32, particles: usize) -> XResult<f64> {
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
                let occupation_time = self.simulate_p()?;
                Ok(occupation_time.powi(order))
            })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / particles as f64;
        Ok(result)
    }

    /// Get the central moment of the occupation time
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    pub fn central_moment_p(&self, order: i32, particles: usize) -> XResult<f64> {
        let mean = self.raw_moment_p(order, particles)?;
        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let occupation_time = self.simulate_p()?;
                Ok((occupation_time - mean).powi(order))
            })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / particles as f64;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::continuous::Bm;
    #[test]
    fn test_first_passage_time() {
        let sp = Bm::default();
        let fpt = FirstPassageTime::new(&sp, (0.0, 1.0)).unwrap();
        let fpt_result = fpt.simulate(1000.0, 0.1).unwrap();
        assert!(fpt_result.is_some());
    }

    #[test]
    fn test_occupation_time() {
        let sp = Bm::default();
        let ot = OccupationTime::new(&sp, (0.0, 1.0), 1000.0).unwrap();
        let ot_result = ot.simulate(0.1).unwrap();
        assert!(ot_result > 0.0);
    }
}
