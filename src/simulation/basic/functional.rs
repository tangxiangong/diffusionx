//! Functional distribution of stochastic processes
//!
//! - First passage time [FirstPassageTime]
//! - Occupation time [OccupationTime]

use crate::{
    FloatExt, SimulationError, XResult,
    simulation::prelude::{ContinuousProcess, PointProcess},
};
use rayon::prelude::*;

/// First passage time
#[derive(Debug, Clone)]
pub struct FirstPassageTime<'a, SP, T: FloatExt = f64> {
    /// The stochastic process
    sp: &'a SP,
    /// The domain that the first passage time is interested in
    domain: (T, T),
}

impl<'a, SP: Send + Sync, T: FloatExt> FirstPassageTime<'a, SP, T> {
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
    /// use diffusionx::simulation::continuous::Bm;
    /// use diffusionx::simulation::functional::FirstPassageTime;
    ///
    /// let sp = Bm::default();
    /// let fpt = FirstPassageTime::new(&sp, (0.0, 1.0)).unwrap();
    /// ```
    pub fn new(sp: &'a SP, domain: (T, T)) -> XResult<Self> {
        if domain.0 >= domain.1 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `domain` must be a valid interval, i.e., `domain.0 < domain.1`, got `{domain:?}`"
            ))
            .into());
        }
        Ok(Self { sp, domain })
    }
}

impl<'a, SP: ContinuousProcess<T>, T: FloatExt> FirstPassageTime<'a, SP, T> {
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
    /// use diffusionx::simulation::continuous::Bm;
    /// use diffusionx::simulation::functional::FirstPassageTime;
    ///
    /// let sp = Bm::default();
    /// let fpt = FirstPassageTime::new(&sp, (0.0, 1.0)).unwrap();
    /// let result = fpt.simulate(1000.0, 0.1).unwrap();
    /// ```
    pub fn simulate(&self, max_duration: T, time_step: T) -> XResult<Option<T>> {
        let ten = T::from(10).unwrap();
        let two = T::from(2).unwrap();

        let (a, b) = self.domain;
        let find = |x: &[T]| x.iter().position(|&pos| pos <= a || pos >= b);
        let mut duration = (max_duration / ten).min(ten);
        loop {
            let (t, x) = self.sp.simulate(duration, time_step)?;
            if let Some(index) = find(&x) {
                return Ok(Some(t[index]));
            }
            duration *= two;
            if duration > max_duration {
                duration = max_duration;
                let (t, x) = self.sp.simulate(duration, time_step)?;
                return if let Some(index) = find(&x) {
                    Ok(Some(t[index]))
                } else {
                    Ok(None)
                };
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
    /// use diffusionx::simulation::continuous::Bm;
    /// use diffusionx::simulation::functional::FirstPassageTime;
    ///
    /// let sp = Bm::default();
    /// let fpt = FirstPassageTime::new(&sp, (0.0, 1.0)).unwrap();
    /// let result = fpt.raw_moment(1, 1000, 1000.0, 0.1).unwrap();
    /// ```
    pub fn raw_moment(
        &self,
        order: i32,
        particles: usize,
        max_duration: T,
        time_step: T,
    ) -> XResult<Option<T>> {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got `{particles}`"
            ))
            .into());
        }
        if order == 0 {
            return Ok(Some(T::one()));
        }
        if time_step <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `time_step` must be positive, got `{time_step:?}`"
            ))
            .into());
        }
        if max_duration <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `max_duration` must be positive, got `{max_duration:?}`"
            ))
            .into());
        }

        // Collect all valid FPT values
        let valid_values = (0..particles)
            .into_par_iter()
            .map(|_| {
                self.simulate(max_duration, time_step)
                    .unwrap()
                    .map(|t| t.powi(order))
            })
            .filter_map(|x| x)
            .collect::<Vec<_>>();

        Ok(average(valid_values))
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
    /// use diffusionx::simulation::continuous::Bm;
    /// use diffusionx::simulation::functional::FirstPassageTime;
    ///
    /// let sp = Bm::default();
    /// let fpt = FirstPassageTime::new(&sp, (0.0, 1.0)).unwrap();
    /// let result = fpt.central_moment(1, 1000, 1000.0, 0.1).unwrap();
    /// ```
    pub fn central_moment(
        &self,
        order: i32,
        particles: usize,
        max_duration: T,
        time_step: T,
    ) -> XResult<Option<T>> {
        if order == 0 {
            return Ok(Some(T::one()));
        }

        if max_duration <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `max_duration` must be positive, got `{max_duration:?}`"
            ))
            .into());
        }
        let mean = self.raw_moment(order, particles, max_duration, time_step)?;
        if mean.is_none() {
            return Ok(None);
        }
        let mean = mean.unwrap();
        let valid_values = (0..particles)
            .into_par_iter()
            .map(|_| {
                self.simulate(max_duration, time_step)
                    .unwrap()
                    .map(|t| (t - mean).powi(order))
            })
            .filter_map(|x| x)
            .collect::<Vec<_>>();

        Ok(average(valid_values))
    }
}

/// Occupation time
#[derive(Debug, Clone)]
pub struct OccupationTime<'a, SP, T: FloatExt = f64> {
    /// The stochastic process
    sp: &'a SP,
    /// The domain that the occupation time is interested in
    domain: (T, T),
    /// The duration of the simulation
    duration: T,
}

impl<'a, SP: Send + Sync, T: FloatExt> OccupationTime<'a, SP, T> {
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
    /// use diffusionx::simulation::continuous::Bm;
    /// use diffusionx::simulation::functional::OccupationTime;
    ///
    /// let sp = Bm::default();
    /// let ot = OccupationTime::new(&sp, (0.0, 1.0), 1000.0).unwrap();
    /// ```
    pub fn new(sp: &'a SP, domain: (T, T), duration: T) -> XResult<Self> {
        if domain.0 >= domain.1 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `domain` must be a valid interval, i.e., `domain.0 < domain.1`, got `{domain:?}`"
            ))
            .into());
        }
        if duration <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got `{duration:?}`"
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

impl<'a, SP: ContinuousProcess<T>, T: FloatExt> OccupationTime<'a, SP, T> {
    /// Simulate the occupation time
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::Bm;
    /// use diffusionx::simulation::functional::OccupationTime;
    ///
    /// let sp = Bm::default();
    /// let ot = OccupationTime::new(&sp, (0.0, 1.0), 1000.0).unwrap();
    /// let result = ot.simulate(0.1).unwrap();
    /// ```
    pub fn simulate(&self, time_step: T) -> XResult<T> {
        if time_step <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `time_step` must be positive, got `{time_step:?}`"
            ))
            .into());
        }
        let (t, x) = self.sp.simulate(self.duration, time_step)?;
        let (a, b) = self.domain;

        let occupation_time = oc(&t, &x, a, b);

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
    /// use diffusionx::simulation::continuous::Bm;
    /// use diffusionx::simulation::functional::OccupationTime;
    ///
    /// let sp = Bm::default();
    /// let ot = OccupationTime::new(&sp, (0.0, 1.0), 1000.0).unwrap();
    /// let result = ot.raw_moment(1, 1000, 0.1).unwrap();
    /// ```
    pub fn raw_moment(&self, order: i32, particles: usize, time_step: T) -> XResult<T> {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got `{particles}`"
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

        let result = (0..particles)
            .into_par_iter()
            .map(|_| {
                let occupation_time = self.simulate(time_step).unwrap();
                occupation_time.powi(order)
            })
            .sum::<T>()
            / T::from(particles).unwrap();
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
    /// use diffusionx::simulation::continuous::Bm;
    /// use diffusionx::simulation::functional::OccupationTime;
    ///
    /// let sp = Bm::default();
    /// let ot = OccupationTime::new(&sp, (0.0, 1.0), 1000.0).unwrap();
    /// let result = ot.central_moment(1, 1000, 0.1).unwrap();
    /// ```
    pub fn central_moment(&self, order: i32, particles: usize, time_step: T) -> XResult<T> {
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
                "The `particles` must be positive, got `{particles}`"
            ))
            .into());
        }
        let mean = self.raw_moment(order, particles, time_step)?;
        let result = (0..particles)
            .into_par_iter()
            .map(|_| {
                let occupation_time = self.simulate(time_step).unwrap();
                (occupation_time - mean).powi(order)
            })
            .sum::<T>()
            / T::from(particles).unwrap();
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
        let max_duration = max_duration.into();
        if max_duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `max_duration` must be positive, got `{max_duration}`"
            ))
            .into());
        }
        let (a, b) = self.domain;
        let find = |x: &[f64]| x.iter().position(|&x| x <= a || x >= b);
        let mut duration = (max_duration / 10.0).min(10.0);
        loop {
            let (t, x) = self.sp.simulate_with_duration(duration)?;
            if let Some(index) = find(&x) {
                return Ok(Some(t[index]));
            }
            duration *= 2.0;
            if duration > max_duration {
                duration = max_duration;
                let (t, x) = self.sp.simulate_with_duration(duration)?;
                return if let Some(index) = find(&x) {
                    Ok(Some(t[index]))
                } else {
                    Ok(None)
                };
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
        let max_duration = max_duration.into();
        if max_duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `max_duration` must be positive, got `{max_duration}`"
            ))
            .into());
        }
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(
                "particles must be positive".to_string(),
            )
            .into());
        }
        if order == 0 {
            return Ok(Some(1.0));
        }

        // 使用元组来同时跟踪总和和有效样本数
        let valid_values = (0..particles)
            .into_par_iter()
            .map(|_| {
                self.simulate_p(max_duration)
                    .unwrap()
                    .map(|t| t.powi(order))
            })
            .filter_map(|x| x)
            .collect::<Vec<_>>();
        Ok(average(valid_values))
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
        if max_duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `max_duration` must be positive, got `{max_duration}`"
            ))
            .into());
        }
        if order == 0 {
            return Ok(Some(1.0));
        }
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(
                "particles must be positive".to_string(),
            )
            .into());
        }
        let mean = self.raw_moment_p(order, particles, max_duration)?;
        if mean.is_none() {
            return Ok(None);
        }
        let mean = mean.unwrap();
        let valid_values = (0..particles)
            .into_par_iter()
            .map(|_| {
                self.simulate_p(max_duration)
                    .unwrap()
                    .map(|t| (t - mean).powi(order))
            })
            .filter_map(|x| x)
            .collect::<Vec<_>>();

        Ok(average(valid_values))
    }
}

impl<'a, SP: PointProcess> OccupationTime<'a, SP> {
    pub fn simulate_p(&self) -> XResult<f64> {
        let (t, x) = self.sp.simulate_with_duration(self.duration)?;
        let (a, b) = self.domain;

        let occupation_time = oc(&t, &x, a, b);

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
        if order == 0 {
            return Ok(1.0);
        }

        let result = (0..particles)
            .into_par_iter()
            .map(|_| {
                let occupation_time = self.simulate_p().unwrap();
                occupation_time.powi(order)
            })
            .sum::<f64>()
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
        if order == 0 {
            return Ok(1.0);
        }
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(
                "particles must be positive".to_string(),
            )
            .into());
        }
        let mean = self.raw_moment_p(order, particles)?;
        let result = (0..particles)
            .into_par_iter()
            .map(|_| {
                let occupation_time = self.simulate_p().unwrap();
                (occupation_time - mean).powi(order)
            })
            .sum::<f64>()
            / particles as f64;
        Ok(result)
    }
}

fn average<T: FloatExt>(values: Vec<T>) -> Option<T> {
    if values.is_empty() {
        None
    } else {
        let count = values.len();
        let sum = values.into_par_iter().sum::<T>();
        Some(sum / T::from(count).unwrap())
    }
}

fn oc<T: FloatExt>(t: &[T], x: &[T], a: T, b: T) -> T {
    x.windows(2)
        .zip(t.windows(2))
        .map(|(x_pair, t_pair)| {
            let in_domain = (a..=b).contains(&x_pair[0]) && (a..=b).contains(&x_pair[1]);
            if in_domain {
                t_pair[1] - t_pair[0]
            } else {
                T::zero()
            }
        })
        .sum()
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
