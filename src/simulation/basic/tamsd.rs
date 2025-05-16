use super::{ContinuousProcess, PointProcess};
use crate::{SimulationError, XResult, utils::flatten_interpolate};
use gauss_quad::GaussLegendre;
use rayon::prelude::*;

/// TAMSD (time-averaged mean-squared displacement)
#[derive(Debug, Clone)]
pub struct TAMSD<SP: Clone> {
    /// The continuous process
    process: SP,
    /// The duration
    duration: f64,
    /// The slag length
    delta: f64,
}

impl<SP: Clone> TAMSD<SP> {
    /// Create a new TAMSD
    ///
    /// # Arguments
    ///
    /// * `process` - The continuous process to calculate the TAMSD of.
    /// * `duration` - The duration of the simulation.
    /// * `delta` - The slag length.
    pub fn new(process: &SP, duration: impl Into<f64>, delta: impl Into<f64>) -> XResult<Self> {
        let duration = duration.into();
        let delta = delta.into();
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {}",
                duration
            ))
            .into());
        }
        if delta <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `delta` must be positive, got {}",
                delta
            ))
            .into());
        }

        if duration < delta {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be greater than `delta`, got duration {} and delta {}",
                duration, delta
            ))
            .into());
        }
        Ok(Self {
            process: process.clone(),
            duration,
            delta,
        })
    }

    /// Get the process
    pub fn process(&self) -> &SP {
        &self.process
    }

    /// Get the duration
    pub fn duration(&self) -> f64 {
        self.duration
    }

    /// Get the slag length
    pub fn delta(&self) -> f64 {
        self.delta
    }
}

impl<SP: ContinuousProcess> TAMSD<SP> {
    /// Simulate the TAMSD
    ///
    /// # Arguments
    ///
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    pub fn simulate(&self, time_step: f64, quad_order: usize) -> XResult<f64> {
        let legendre_quad = GaussLegendre::new(quad_order)?;
        let nodes_weights_pairs = legendre_quad.into_node_weight_pairs();
        let duration = self.duration;
        let slag = self.delta;
        let nodes_weights = nodes_weights_transform(0.0, duration - slag, &nodes_weights_pairs);
        let sp = self.process.clone();
        let result = nodes_weights
            .into_par_iter()
            .map(|(node, weight)| -> XResult<f64> {
                let slag_length = (slag / time_step).ceil() as usize;
                let (_, x) = sp.simulate(node + slag, time_step)?;
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
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
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
    pub fn mean(&self, particles: usize, time_step: f64, quad_order: usize) -> XResult<f64> {
        Ok((0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> { self.simulate(time_step, quad_order) })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / particles as f64)
    }

    /// Get the variance of the TAMSD
    ///
    /// # Arguments
    ///
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    /// * `quad_order` - The order of the Gauss-Legendre quadrature.
    pub fn variance(&self, particles: usize, time_step: f64, quad_order: usize) -> XResult<f64> {
        let mean = self.mean(particles, time_step, quad_order)?;
        Ok((0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let value = self.simulate(time_step, quad_order)?;
                Ok((value - mean).powi(2))
            })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / particles as f64)
    }
}

impl<SP: PointProcess> TAMSD<SP> {
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
        let nodes_weights = nodes_weights_transform(0.0, duration - slag, &nodes_weights_pairs);
        let sp = self.process.clone();
        let result = nodes_weights
            .into_par_iter()
            .map(|(node, weight)| -> XResult<f64> {
                let slag_length = (slag / time_step).ceil() as usize;
                let (t, x) = sp.simulate_with_duration(node + slag)?;
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
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
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
        Ok((0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> { self.simulate_p(time_step, quad_order) })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
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
        let mean = self.mean_p(particles, time_step, quad_order)?;
        Ok((0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let value = self.simulate_p(time_step, quad_order)?;
                Ok((value - mean).powi(2))
            })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
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
fn nodes_weights_transform(
    a: impl Into<f64>,
    b: impl Into<f64>,
    pairs: &[(f64, f64)],
) -> Vec<(f64, f64)> {
    let a: f64 = a.into();
    let b: f64 = b.into();
    pairs
        .iter()
        .map(|(node, weight)| {
            let new_weight = weight * (b - a) / 2.0;
            let new_node = (b - a) * node / 2.0 + (b + a) / 2.0;
            (new_node, new_weight)
        })
        .collect()
}
