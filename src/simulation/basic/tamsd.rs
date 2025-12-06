use super::{ContinuousProcess, PointProcess};
use crate::{SimulationError, XResult, utils::flatten_interpolate};
use gauss_quad::GaussLegendre;
use num_traits::Float;
use rayon::prelude::*;
use std::fmt::Debug;

/// TAMSD (time-averaged mean-squared displacement)
#[derive(Debug, Clone)]
pub struct TAMSD<'a, SP, T: Float = f64> {
    /// The continuous process
    process: &'a SP,
    /// The duration
    duration: T,
    /// The slag length
    delta: T,
}

impl<'a, SP, T: Float + Debug> TAMSD<'a, SP, T> {
    /// Create a new TAMSD
    ///
    /// # Arguments
    ///
    /// * `process` - The continuous process to calculate the TAMSD of.
    /// * `duration` - The duration of the simulation.
    /// * `delta` - The slag length.
    pub fn new(process: &'a SP, duration: T, delta: T) -> XResult<Self> {
        if duration <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {duration:?}"
            ))
            .into());
        }
        if delta <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `delta` must be positive, got {delta:?}"
            ))
            .into());
        }

        if duration < delta {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be greater than `delta`, got duration {duration:?} and delta {delta:?}"
            ))
            .into());
        }
        Ok(Self {
            process,
            duration,
            delta,
        })
    }

    /// Get the process
    pub fn get_process(&self) -> &'a SP {
        self.process
    }

    /// Get the duration
    pub fn get_duration(&self) -> T {
        self.duration
    }

    /// Get the slag length
    pub fn get_delta(&self) -> T {
        self.delta
    }
}

impl<'a, SP: ContinuousProcess<T>, T: Float + Debug> TAMSD<'a, SP, T> {
    /// Simulate the TAMSD
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    pub fn simulate(&self, time_step: T, quad_order: usize) -> XResult<T>
    where
        T: Send + Sync + std::iter::Sum,
    {
        let legendre_quad = GaussLegendre::new(quad_order)?;
        let nodes_weights_pairs = legendre_quad.into_node_weight_pairs();
        let duration = self.duration;
        let slag = self.delta;
        let nodes_weights =
            nodes_weights_transform(T::zero(), duration - slag, nodes_weights_pairs);
        let result = nodes_weights
            .into_par_iter()
            .map(|(node, weight)| -> XResult<T> {
                let slag_length = (slag / time_step).ceil().to_usize().unwrap();
                let (_, x) = self.process.simulate(node + slag, time_step)?;
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
            .collect::<XResult<Vec<T>>>()?
            .into_par_iter()
            .sum::<T>()
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
    pub fn mean(&self, particles: usize, time_step: T, quad_order: usize) -> XResult<T>
    where
        T: Send + Sync + std::iter::Sum,
    {
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

        if quad_order == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `quad_order` must be positive, got `{quad_order}`"
            ))
            .into());
        }

        Ok((0..particles)
            .into_par_iter()
            .map(|_| self.simulate(time_step, quad_order).unwrap())
            .sum::<T>()
            / T::from(particles).unwrap())
    }

    /// Get the variance of the TAMSD
    ///
    /// # Arguments
    ///
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    pub fn variance(&self, particles: usize, time_step: T, quad_order: usize) -> XResult<T>
    where
        T: Send + Sync + std::iter::Sum,
    {
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
        if quad_order == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `quad_order` must be positive, got `{quad_order}`"
            ))
            .into());
        }
        let mean = self.mean(particles, time_step, quad_order)?;
        Ok((0..particles)
            .into_par_iter()
            .map(|_| {
                let value = self.simulate(time_step, quad_order).unwrap();
                (value - mean).powi(2)
            })
            .into_par_iter()
            .sum::<T>()
            / T::from(particles).unwrap())
    }
}

impl<'a, SP: PointProcess> TAMSD<'a, SP> {
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
        let nodes_weights = nodes_weights_transform(0.0, duration - slag, nodes_weights_pairs);
        let result = nodes_weights
            .into_par_iter()
            .map(|(node, weight)| -> XResult<f64> {
                let slag_length = (slag / time_step).ceil() as usize;
                let (t, x) = self.process.simulate_with_duration(node + slag)?;
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
            .collect::<XResult<Vec<_>>>()?
            .into_par_iter()
            .sum::<f64>()
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
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {particles}"
            ))
            .into());
        }
        if time_step <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `time_step` must be positive, got `{time_step}`"
            ))
            .into());
        }
        if quad_order == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `quad_order` must be positive, got `{quad_order}`"
            ))
            .into());
        }
        Ok((0..particles)
            .into_par_iter()
            .map(|_| self.simulate_p(time_step, quad_order).unwrap())
            .sum::<f64>()
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
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `particles` must be positive, got {particles}"
            ))
            .into());
        }
        if time_step <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `time_step` must be positive, got `{time_step}`"
            ))
            .into());
        }
        if quad_order == 0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `quad_order` must be positive, got `{quad_order}`"
            ))
            .into());
        }
        let mean = self.mean_p(particles, time_step, quad_order)?;
        Ok((0..particles)
            .into_par_iter()
            .map(|_| {
                let value = self.simulate_p(time_step, quad_order).unwrap();
                (value - mean).powi(2)
            })
            .sum::<f64>()
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
fn nodes_weights_transform<T: Float>(a: T, b: T, pairs: Vec<(f64, f64)>) -> Vec<(T, T)> {
    let two = T::from(2).unwrap();
    pairs
        .into_iter()
        .map(|(node, weight)| {
            let new_weight = T::from(weight).unwrap() * (b - a) / two;
            let new_node = (b - a) * T::from(node).unwrap() / two + (b + a) / two;
            (new_node, new_weight)
        })
        .collect()
}
