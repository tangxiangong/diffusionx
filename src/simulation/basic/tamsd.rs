use std::num::NonZeroUsize;

use crate::{
    FloatExt, RealExt, SimulationError, XResult, simulation::prelude::*, utils::flatten_interpolate,
};
use gauss_quad::GaussLegendre;
use rayon::prelude::*;

/// Time-averaged mean-squared displacement estimator.
///
/// For a trajectory over duration \(T\) and lag \(\Delta\), TAMSD is
///
/// $$\overline{\delta^2(\Delta; T)}
/// = \frac{1}{T-\Delta}\int_0^{T-\Delta}
/// \left[X(t+\Delta)-X(t)\right]^2\,dt.$$
#[derive(Debug, Clone)]
pub struct TAMSD<'a, SP, T: FloatExt = f64> {
    /// The continuous process
    process: &'a SP,
    /// The duration
    duration: T,
    /// The lag length
    delta: T,
}

impl<'a, SP, T: FloatExt> TAMSD<'a, SP, T> {
    /// Create a new TAMSD estimator.
    ///
    /// # Arguments
    ///
    /// * `process` - The continuous process to calculate the TAMSD of.
    /// * `duration` - The duration of the simulation.
    /// * `delta` - The lag length.
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

    /// Get the process being sampled.
    pub fn get_process(&self) -> &'a SP {
        self.process
    }

    /// Get the trajectory duration.
    pub fn get_duration(&self) -> T {
        self.duration
    }

    /// Get the lag length.
    pub fn get_delta(&self) -> T {
        self.delta
    }
}

impl<'a, SP: ContinuousProcess<T>, T: FloatExt> TAMSD<'a, SP, T> {
    /// Estimate the TAMSD for one continuous-process trajectory.
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    pub fn simulate(&self, time_step: T, quad_order: usize) -> XResult<T> {
        let quad_order =
            NonZeroUsize::new(quad_order).unwrap_or_else(|| NonZeroUsize::new(10).unwrap());
        let legendre_quad = GaussLegendre::new(quad_order);
        let nodes_weights_pairs = legendre_quad.into_node_weight_pairs().to_vec();
        let duration = self.duration;
        let slag = self.delta;
        let nodes_weights =
            nodes_weights_transform(T::zero(), duration - slag, nodes_weights_pairs);
        let sum = nodes_weights
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
            .try_reduce(T::zero, |a, b| Ok::<T, crate::XError>(a + b))?;
        let result = sum / (duration - slag);
        Ok(result)
    }

    /// Estimate the ensemble average of the TAMSD for a continuous process.
    ///
    /// # Arguments
    ///
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    pub fn mean(&self, particles: usize, time_step: T, quad_order: usize) -> XResult<T> {
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

        let sum = (0..particles)
            .into_par_iter()
            .map(|_| self.simulate(time_step, quad_order))
            .try_reduce(T::zero, |a, b| Ok::<T, crate::XError>(a + b))?;
        Ok(sum / T::from(particles).unwrap())
    }

    /// Estimate the ensemble variance of the TAMSD for a continuous process.
    ///
    /// # Arguments
    ///
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    pub fn variance(&self, particles: usize, time_step: T, quad_order: usize) -> XResult<T> {
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
        let sum = (0..particles)
            .into_par_iter()
            .map(|_| {
                let value = self.simulate(time_step, quad_order)?;
                Ok::<T, crate::XError>((value - mean).powi(2))
            })
            .try_reduce(T::zero, |a, b| Ok::<T, crate::XError>(a + b))?;
        Ok(sum / T::from(particles).unwrap())
    }
}

impl<'a, SP, T: FloatExt> TAMSD<'a, SP, T> {
    /// Estimate the TAMSD for one point-process trajectory.
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    pub fn simulate_p<X: RealExt>(&self, time_step: T, quad_order: usize) -> XResult<f64>
    where
        SP: PointProcess<T, X> + Clone,
    {
        let quad_order =
            NonZeroUsize::new(quad_order).unwrap_or_else(|| NonZeroUsize::new(10).unwrap());
        let legendre_quad = GaussLegendre::new(quad_order);
        let nodes_weights_pairs = legendre_quad.into_node_weight_pairs().to_vec();
        let duration = self.duration;
        let slag = self.delta;
        let nodes_weights = nodes_weights_transform(
            0.0,
            (duration - slag).to_f64().unwrap(),
            nodes_weights_pairs,
        );
        let sum = nodes_weights
            .into_par_iter()
            .map(|(node, weight)| -> XResult<f64> {
                let slag_length = (slag / time_step).ceil().to_usize().unwrap();
                let (t, x) = self
                    .process
                    .simulate_with_duration(T::from(node).unwrap() + slag)?;
                let (_, x) = flatten_interpolate(&t, &x, time_step)?;
                let len = x.len();
                let end_position = x.last();
                let slag_position = x.get(len - slag_length - 1);
                if end_position.is_none() || slag_position.is_none() {
                    return Err(SimulationError::Unknown.into());
                }
                let end_position = end_position.unwrap().to_f64().unwrap();
                let slag_position = slag_position.unwrap().to_f64().unwrap();

                Ok((end_position - slag_position).powi(2) * weight)
            })
            .try_reduce(|| 0.0, |a, b| Ok::<f64, crate::XError>(a + b))?;
        let result = sum / (duration - slag).to_f64().unwrap();
        Ok(result)
    }

    /// Estimate the ensemble average of the TAMSD for a point process.
    ///
    /// # Arguments
    ///
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    pub fn mean_p<X: RealExt>(
        &self,
        particles: usize,
        time_step: T,
        quad_order: usize,
    ) -> XResult<f64>
    where
        SP: PointProcess<T, X> + Clone,
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
        let sum = (0..particles)
            .into_par_iter()
            .map(|_| self.simulate_p(time_step, quad_order))
            .try_reduce(|| 0.0, |a, b| Ok::<f64, crate::XError>(a + b))?;
        Ok(sum / particles as f64)
    }

    /// Estimate the ensemble variance of the TAMSD for a point process.
    ///
    /// # Arguments
    ///
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    pub fn variance_p<X: RealExt>(
        &self,
        particles: usize,
        time_step: T,
        quad_order: usize,
    ) -> XResult<f64>
    where
        SP: PointProcess<T, X> + Clone,
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
        let mean = self.mean_p(particles, time_step, quad_order)?;
        let sum = (0..particles)
            .into_par_iter()
            .map(|_| {
                let value = self.simulate_p(time_step, quad_order)?;
                Ok::<f64, crate::XError>((value - mean).powi(2))
            })
            .try_reduce(|| 0.0, |a, b| Ok::<f64, crate::XError>(a + b))?;
        Ok(sum / particles as f64)
    }
}

/// Transform the nodes and weights
///
/// # Arguments
///
/// * `a` - The lower bound of the interval.
/// * `b` - The upper bound of the interval.
/// * `pairs` - The nodes and weights pairs of the unit interval.
fn nodes_weights_transform<T: FloatExt>(a: T, b: T, pairs: Vec<(f64, f64)>) -> Vec<(T, T)> {
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
