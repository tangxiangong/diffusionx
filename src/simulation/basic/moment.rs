use crate::{FloatExt, RealExt, SimulationError, XResult, simulation::prelude::*};
use rayon::prelude::*;

/// Estimators for ensemble moments of simulated terminal positions.
///
/// Implementations repeatedly simulate independent particles and average the
/// requested statistic. For a terminal value \(X_T\) and particle count \(N\), the
/// raw moment estimator is
///
/// $$\mathbb{E}\!\left(X_T^k\right) \approx \frac{1}{N}\sum_{i=1}^{N} X_T^{(i)k}.$$
pub trait Moment<T: FloatExt = f64> {
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

    /// Get the mean
    fn mean(&self, particles: usize, time_step: T) -> XResult<T> {
        self.raw_moment(1, particles, time_step)
    }

    /// Get the mean square displacement of the simulation.
    ///
    /// The estimator averages squared displacements over independent particles:
    ///
    /// $$\operatorname{MSD}(T) \approx
    /// \frac{1}{N}\sum_{i=1}^{N}\left(X_T^{(i)} - X_0^{(i)}\right)^2.$$
    fn msd(&self, particles: usize, time_step: T) -> XResult<T>;

    /// Get the fractional raw absolute moment of the simulation.
    ///
    /// This estimates \(\mathbb{E}\!\left(|X_T|^q\right)\) for a real-valued order \(q\):
    ///
    /// $$\mathbb{E}\!\left(|X_T|^q\right) \approx \frac{1}{N}\sum_{i=1}^{N}|X_T^{(i)}|^q.$$
    fn frac_raw_moment(&self, order: T, particles: usize, time_step: T) -> XResult<T>;

    /// Get the fractional central absolute moment of the simulation.
    ///
    /// This estimates \(\mathbb{E}\!\left(|X_T - \mathbb{E}(X_T)|^q\right)\) for a real-valued order \(q\):
    ///
    /// $$\mathbb{E}\!\left(|X_T-\mathbb{E}(X_T)|^q\right) \approx
    /// \frac{1}{N}\sum_{i=1}^{N}|X_T^{(i)}-\bar{X}_T|^q.$$
    fn frac_central_moment(&self, order: T, particles: usize, time_step: T) -> XResult<T>;
}

impl<T: FloatExt, SP: ContinuousProcess<T> + Clone> Moment<T> for ContinuousTrajectory<SP, T> {
    fn msd(&self, particles: usize, time_step: T) -> XResult<T> {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {particles}"
            ))
            .into());
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
                let delta_x = match self.sp.displacement(duration, time_step) {
                    Ok(delta_x) => delta_x,
                    Err(e) => panic!("{}", e),
                };

                delta_x * delta_x
            })
            .sum::<T>();

        let result = values / (T::from(particles).unwrap());
        Ok(result)
    }

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

    fn frac_raw_moment(&self, order: T, particles: usize, time_step: T) -> XResult<T> {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {particles}"
            ))
            .into());
        }

        if order == T::zero() {
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
                if order == T::one() {
                    end.abs()
                } else {
                    end.abs().powf(order)
                }
            })
            .sum::<T>();

        let result = values / (T::from(particles).unwrap());
        Ok(result)
    }

    fn frac_central_moment(&self, order: T, particles: usize, time_step: T) -> XResult<T> {
        if order == T::zero() {
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
                if order == T::one() {
                    (end_position - mean).abs()
                } else {
                    (end_position - mean).abs().powf(order)
                }
            })
            .sum::<T>();

        let result = values / (T::from(particles).unwrap());
        Ok(result)
    }
}

impl<SP: DiscreteProcess<N, X> + Clone, N: IntExt, X: RealExt> Moment
    for DiscreteTrajectory<SP, N, X>
{
    fn msd(&self, particles: usize, _: f64) -> XResult<f64> {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {particles}"
            ))
            .into());
        }

        let num_step = self.num_step;

        let values = (0..particles)
            .into_par_iter()
            .map(|_| {
                let delta_x = match self.sp.displacement(num_step) {
                    Ok(delta_x) => delta_x.to_f64().unwrap(),
                    Err(e) => panic!("{}", e),
                };
                delta_x * delta_x
            })
            .sum::<f64>();

        let result = values / (particles as f64);
        Ok(result)
    }

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
                    Ok(end_position) => end_position.to_f64().unwrap(),
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
                    Ok(end_position) => end_position.to_f64().unwrap(),
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

    fn frac_raw_moment(&self, order: f64, particles: usize, _: f64) -> XResult<f64> {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {particles}"
            ))
            .into());
        }

        if order == 0.0 {
            return Ok(1.0);
        }

        let num_step = self.num_step;

        let values = (0..particles)
            .into_par_iter()
            .map(|_| {
                let end_position = match self.sp.end(num_step) {
                    Ok(end_position) => end_position.to_f64().unwrap(),
                    Err(e) => panic!("{}", e),
                };
                if order == 1.0 {
                    end_position.abs()
                } else {
                    end_position.abs().powf(order)
                }
            })
            .sum::<f64>();

        let result = values / (particles as f64);
        Ok(result)
    }

    fn frac_central_moment(&self, order: f64, particles: usize, _: f64) -> XResult<f64> {
        if order == 0.0 {
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
                    Ok(end_position) => end_position.to_f64().unwrap(),
                    Err(e) => panic!("{}", e),
                };
                if order == 1.0 {
                    (end_position - mean).abs()
                } else {
                    (end_position - mean).abs().powf(order)
                }
            })
            .sum::<f64>();

        let result = values / (particles as f64);
        Ok(result)
    }
}

impl<SP: PointProcess<T, X> + Clone, T: FloatExt, X: RealExt> Moment for PointTrajectory<SP, T, X> {
    fn msd(&self, particles: usize, _: f64) -> XResult<f64> {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {particles}"
            ))
            .into());
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
                let delta_x = match self.sp.displacement(duration) {
                    Ok(delta_x) => delta_x.to_f64().unwrap(),
                    Err(e) => panic!("{}", e),
                };
                delta_x * delta_x
            })
            .sum::<f64>();

        let result = values / (particles as f64);
        Ok(result)
    }

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
                    Ok(end_position) => end_position.to_f64().unwrap(),
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
                    Ok(end_position) => end_position.to_f64().unwrap(),
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

    fn frac_raw_moment(&self, order: f64, particles: usize, _: f64) -> XResult<f64> {
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {particles}"
            ))
            .into());
        }

        if order == 0.0 {
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
                    Ok(end_position) => end_position.to_f64().unwrap(),
                    Err(e) => panic!("{}", e),
                };
                if order == 1.0 {
                    end_position.abs()
                } else {
                    end_position.abs().powf(order)
                }
            })
            .sum::<f64>();

        let result = values / (particles as f64);
        Ok(result)
    }

    fn frac_central_moment(&self, order: f64, particles: usize, _time_step: f64) -> XResult<f64> {
        if order == 0.0 {
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
                    Ok(end_position) => end_position.to_f64().unwrap(),
                    Err(e) => panic!("{}", e),
                };
                if order == 1.0 {
                    (end_position - mean).abs()
                } else {
                    (end_position - mean).abs().powf(order)
                }
            })
            .sum::<f64>();

        let result = values / (particles as f64);
        Ok(result)
    }
}
