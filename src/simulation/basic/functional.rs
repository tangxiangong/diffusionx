//! Functional distribution of stochastic processes
//!
//! - First passage time [FirstPassageTime]
//! - Occupation time [OccupationTime]

use crate::{
    FloatExt, RealExt, SimulationError, XResult,
    simulation::prelude::{ContinuousProcess, PointProcess},
};
use rayon::prelude::*;

/// First passage time
#[derive(Debug, Clone)]
pub struct FirstPassageTime<'a, SP, T: RealExt = f64> {
    /// The stochastic process
    sp: &'a SP,
    /// The domain that the first passage time is interested in
    domain: (T, T),
}

impl<'a, SP: Send + Sync, T: RealExt> FirstPassageTime<'a, SP, T> {
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
    /// use diffusionx::simulation::basic::FirstPassageTime;
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
    /// use diffusionx::simulation::basic::FirstPassageTime;
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
    /// use diffusionx::simulation::basic::FirstPassageTime;
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

        let (sum, count) = (0..particles)
            .into_par_iter()
            .try_fold(
                || (T::zero(), 0usize),
                |(sum, count), _| -> XResult<(T, usize)> {
                    Ok(match self.simulate(max_duration, time_step)? {
                        Some(t) => (sum + t.powi(order), count + 1),
                        None => (sum, count),
                    })
                },
            )
            .try_reduce(
                || (T::zero(), 0usize),
                |(sum_a, count_a), (sum_b, count_b)| {
                    Ok::<(T, usize), crate::XError>((sum_a + sum_b, count_a + count_b))
                },
            )?;

        Ok(average_from_sum_count(sum, count))
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
    /// use diffusionx::simulation::basic::FirstPassageTime;
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
        let mean = self.raw_moment(1, particles, max_duration, time_step)?;
        if mean.is_none() {
            return Ok(None);
        }
        let mean = mean.unwrap();
        let (sum, count) = (0..particles)
            .into_par_iter()
            .try_fold(
                || (T::zero(), 0usize),
                |(sum, count), _| -> XResult<(T, usize)> {
                    Ok(match self.simulate(max_duration, time_step)? {
                        Some(t) => (sum + (t - mean).powi(order), count + 1),
                        None => (sum, count),
                    })
                },
            )
            .try_reduce(
                || (T::zero(), 0usize),
                |(sum_a, count_a), (sum_b, count_b)| {
                    Ok::<(T, usize), crate::XError>((sum_a + sum_b, count_a + count_b))
                },
            )?;

        Ok(average_from_sum_count(sum, count))
    }
}

/// Occupation time
#[derive(Debug, Clone)]
pub struct OccupationTime<'a, SP, T: FloatExt = f64, X: RealExt = T> {
    /// The stochastic process
    sp: &'a SP,
    /// The domain that the occupation time is interested in
    domain: (X, X),
    /// The duration of the simulation
    duration: T,
}

impl<'a, SP: Send + Sync, T: FloatExt, X: RealExt> OccupationTime<'a, SP, T, X> {
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
    /// use diffusionx::simulation::basic::OccupationTime;
    ///
    /// let sp = Bm::default();
    /// let ot = OccupationTime::new(&sp, (0.0, 1.0), 1000.0).unwrap();
    /// ```
    pub fn new(sp: &'a SP, domain: (X, X), duration: T) -> XResult<Self> {
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
    /// use diffusionx::simulation::basic::OccupationTime;
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
    /// use diffusionx::simulation::basic::OccupationTime;
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

        let sum = (0..particles)
            .into_par_iter()
            .map(|_| {
                let occupation_time = self.simulate(time_step)?;
                Ok::<T, crate::XError>(occupation_time.powi(order))
            })
            .try_reduce(T::zero, |a, b| Ok::<T, crate::XError>(a + b))?;
        Ok(sum / T::from(particles).unwrap())
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
    /// use diffusionx::simulation::basic::OccupationTime;
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
        let mean = self.raw_moment(1, particles, time_step)?;
        let sum = (0..particles)
            .into_par_iter()
            .map(|_| {
                let occupation_time = self.simulate(time_step)?;
                Ok::<T, crate::XError>((occupation_time - mean).powi(order))
            })
            .try_reduce(T::zero, |a, b| Ok::<T, crate::XError>(a + b))?;
        Ok(sum / T::from(particles).unwrap())
    }
}

impl<'a, SP, X: RealExt> FirstPassageTime<'a, SP, X> {
    /// Simulate the first passage time
    ///
    /// # Arguments
    ///
    /// * `max_duration` - The maximum duration of the simulation.
    pub fn simulate_p<T: FloatExt>(&self, max_duration: T) -> XResult<Option<T>>
    where
        SP: PointProcess<T, X> + Clone,
    {
        if max_duration <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `max_duration` must be positive, got `{max_duration:?}`"
            ))
            .into());
        }
        let (a, b) = self.domain;
        let find = |x: &[X]| x.iter().position(|&x| x <= a || x >= b);
        let ten = T::from(10).unwrap();
        let two = T::from(2).unwrap();
        let mut duration = (max_duration / ten).min(ten);
        loop {
            let (t, x) = self.sp.simulate_with_duration(duration)?;
            if let Some(index) = find(&x) {
                return Ok(Some(t[index]));
            }
            duration *= two;
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
    pub fn raw_moment_p<T: FloatExt>(
        &self,
        order: i32,
        particles: usize,
        max_duration: T,
    ) -> XResult<Option<f64>>
    where
        SP: PointProcess<T, X> + Clone,
    {
        if max_duration <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `max_duration` must be positive, got `{max_duration:?}`"
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

        let (sum, count) = (0..particles)
            .into_par_iter()
            .try_fold(
                || (0.0, 0usize),
                |(sum, count), _| -> XResult<(f64, usize)> {
                    Ok(match self.simulate_p(max_duration)? {
                        Some(t) => (sum + t.to_f64().unwrap().powi(order), count + 1),
                        None => (sum, count),
                    })
                },
            )
            .try_reduce(
                || (0.0, 0usize),
                |(sum_a, count_a), (sum_b, count_b)| {
                    Ok::<(f64, usize), crate::XError>((sum_a + sum_b, count_a + count_b))
                },
            )?;
        Ok(average_from_sum_count(sum, count))
    }

    /// Get the central moment of the first passage time
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `max_duration` - The maximum duration of the simulation.
    pub fn central_moment_p<T: FloatExt>(
        &self,
        order: i32,
        particles: usize,
        max_duration: T,
    ) -> XResult<Option<f64>>
    where
        SP: PointProcess<T, X> + Clone,
    {
        if max_duration <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `max_duration` must be positive, got `{max_duration:?}`"
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
        let mean = self.raw_moment_p(1, particles, max_duration)?;
        if mean.is_none() {
            return Ok(None);
        }
        let mean = mean.unwrap();
        let (sum, count) = (0..particles)
            .into_par_iter()
            .try_fold(
                || (0.0, 0usize),
                |(sum, count), _| -> XResult<(f64, usize)> {
                    Ok(match self.simulate_p(max_duration)? {
                        Some(t) => (sum + (t.to_f64().unwrap() - mean).powi(order), count + 1),
                        None => (sum, count),
                    })
                },
            )
            .try_reduce(
                || (0.0, 0usize),
                |(sum_a, count_a), (sum_b, count_b)| {
                    Ok::<(f64, usize), crate::XError>((sum_a + sum_b, count_a + count_b))
                },
            )?;

        Ok(average_from_sum_count(sum, count))
    }
}

impl<'a, SP: PointProcess<T, X>, T: FloatExt, X: RealExt> OccupationTime<'a, SP, T, X> {
    /// Simulate the occupation time for a point process.
    ///
    /// The occupation time is the total amount of simulated time spent inside the
    /// configured domain:
    ///
    /// $$A_T = \int_0^T \mathbf{1}_{\{a \le X(t) \le b\}}\,dt.$$
    pub fn simulate_p(&self) -> XResult<T> {
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

        let sum = (0..particles)
            .into_par_iter()
            .map(|_| {
                let occupation_time = self.simulate_p()?;
                Ok::<f64, crate::XError>(occupation_time.to_f64().unwrap().powi(order))
            })
            .try_reduce(|| 0.0, |a, b| Ok::<f64, crate::XError>(a + b))?;
        Ok(sum / particles as f64)
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
        let mean = self.raw_moment_p(1, particles)?;
        let sum = (0..particles)
            .into_par_iter()
            .map(|_| {
                let occupation_time = self.simulate_p()?;
                Ok::<f64, crate::XError>((occupation_time.to_f64().unwrap() - mean).powi(order))
            })
            .try_reduce(|| 0.0, |a, b| Ok::<f64, crate::XError>(a + b))?;
        Ok(sum / particles as f64)
    }
}

fn average_from_sum_count<T: FloatExt>(sum: T, count: usize) -> Option<T> {
    if count == 0 {
        None
    } else {
        Some(sum / T::from(count).unwrap())
    }
}

fn oc<T: RealExt, V: FloatExt>(t: &[V], x: &[T], a: T, b: T) -> V {
    x.windows(2)
        .zip(t.windows(2))
        .map(|(x_pair, t_pair)| {
            let in_domain = (a..=b).contains(&x_pair[0]) && (a..=b).contains(&x_pair[1]);
            if in_domain {
                t_pair[1] - t_pair[0]
            } else {
                V::zero()
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::continuous::Bm;

    struct DeterministicExit;

    impl ContinuousProcess<f64> for DeterministicExit {
        fn simulate(&self, duration: f64, _: f64) -> XResult<(Vec<f64>, Vec<f64>)> {
            if duration < 0.5 {
                Ok((vec![0.0, duration], vec![0.0, 0.0]))
            } else {
                Ok((vec![0.0, 0.5, duration], vec![0.0, 2.0, 2.0]))
            }
        }

        fn start(&self) -> f64 {
            0.0
        }
    }

    #[derive(Clone)]
    struct DeterministicPointExit;

    impl PointProcess<f64, f64> for DeterministicPointExit {
        fn start(&self) -> f64 {
            0.0
        }

        fn simulate_with_step(&self, num_step: usize) -> XResult<(Vec<f64>, Vec<f64>)> {
            let t = (0..=num_step).map(|i| i as f64 * 0.5).collect::<Vec<_>>();
            let x = t
                .iter()
                .map(|&time| if time >= 0.5 { 2.0 } else { 0.0 })
                .collect::<Vec<_>>();
            Ok((t, x))
        }

        fn simulate_with_duration(&self, duration: f64) -> XResult<(Vec<f64>, Vec<f64>)> {
            if duration < 0.5 {
                Ok((vec![0.0, duration], vec![0.0, 0.0]))
            } else {
                Ok((vec![0.0, 0.5, duration], vec![0.0, 2.0, 2.0]))
            }
        }
    }

    struct AlwaysInside;

    impl ContinuousProcess<f64> for AlwaysInside {
        fn simulate(&self, duration: f64, _: f64) -> XResult<(Vec<f64>, Vec<f64>)> {
            Ok((vec![0.0, duration], vec![0.0, 0.0]))
        }

        fn start(&self) -> f64 {
            0.0
        }
    }

    #[derive(Clone)]
    struct AlwaysInsidePoint;

    impl PointProcess<f64, f64> for AlwaysInsidePoint {
        fn start(&self) -> f64 {
            0.0
        }

        fn simulate_with_step(&self, num_step: usize) -> XResult<(Vec<f64>, Vec<f64>)> {
            let t = (0..=num_step).map(|i| i as f64).collect::<Vec<_>>();
            let x = vec![0.0; num_step + 1];
            Ok((t, x))
        }

        fn simulate_with_duration(&self, duration: f64) -> XResult<(Vec<f64>, Vec<f64>)> {
            Ok((vec![0.0, duration], vec![0.0, 0.0]))
        }
    }

    #[test]
    fn test_first_passage_time() {
        let sp = Bm::default();
        let fpt = FirstPassageTime::new(&sp, (0.0, 1.0)).unwrap();
        let fpt_result = fpt.simulate(1000.0, 0.1).unwrap();
        assert!(fpt_result.is_some());
    }

    #[test]
    fn test_first_passage_central_moment_uses_mean() {
        let sp = DeterministicExit;
        let fpt = FirstPassageTime::new(&sp, (-1.0, 1.0)).unwrap();
        let moment = fpt.central_moment(2, 4, 2.0, 0.5).unwrap().unwrap();
        assert_eq!(moment, 0.0);
    }

    #[test]
    fn test_point_first_passage_central_moment_uses_mean() {
        let sp = DeterministicPointExit;
        let fpt = FirstPassageTime::new(&sp, (-1.0, 1.0)).unwrap();
        let moment = fpt.central_moment_p(2, 4, 2.0).unwrap().unwrap();
        assert_eq!(moment, 0.0);
    }

    #[test]
    fn test_occupation_time() {
        let sp = Bm::default();
        let ot = OccupationTime::new(&sp, (0.0, 1.0), 1000.0).unwrap();
        let ot_result = ot.simulate(0.1).unwrap();
        assert!(ot_result > 0.0);
    }

    #[test]
    fn test_occupation_central_moment_uses_mean() {
        let sp = AlwaysInside;
        let ot = OccupationTime::new(&sp, (-1.0, 1.0), 2.0).unwrap();
        let moment = ot.central_moment(2, 4, 1.0).unwrap();
        assert_eq!(moment, 0.0);
    }

    #[test]
    fn test_point_occupation_central_moment_uses_mean() {
        let sp = AlwaysInsidePoint;
        let ot = OccupationTime::new(&sp, (-1.0, 1.0), 2.0).unwrap();
        let moment = ot.central_moment_p(2, 4).unwrap();
        assert_eq!(moment, 0.0);
    }
}
