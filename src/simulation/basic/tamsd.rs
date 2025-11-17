use super::{ContinuousProcess, PointProcess};
use crate::{SimulationError, XResult, utils::flatten_interpolate};
use gauss_quad::GaussLegendre;
use rayon::prelude::*;

/// TAMSD (time-averaged mean-squared displacement)
#[derive(Debug, Clone)]
pub struct TAMSD<'a, SP> {
    /// The continuous process
    process: &'a SP,
    /// The duration
    duration: f64,
    /// The slag length
    delta: f64,
}

impl<'a, SP: ContinuousProcess> TAMSD<'a, SP> {
    /// Create a new TAMSD
    ///
    /// # Arguments
    ///
    /// * `process` - The continuous process to calculate the TAMSD of.
    /// * `duration` - The duration of the simulation.
    /// * `delta` - The slag length.
    pub fn new(process: &'a SP, duration: impl Into<f64>, delta: impl Into<f64>) -> XResult<Self> {
        let duration = duration.into();
        let delta = delta.into();
        if duration <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be positive, got {duration}"
            ))
            .into());
        }
        if delta <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `delta` must be positive, got {delta}"
            ))
            .into());
        }

        if duration < delta {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be greater than `delta`, got duration {duration} and delta {delta}"
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
    pub fn get_duration(&self) -> f64 {
        self.duration
    }

    /// Get the slag length
    pub fn get_delta(&self) -> f64 {
        self.delta
    }
}

impl<'a, SP: ContinuousProcess> TAMSD<'a, SP> {
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
        let result = nodes_weights
            .into_par_iter()
            .map(|(node, weight)| -> XResult<f64> {
                let slag_length = (slag / time_step).ceil() as usize;
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
    pub fn mean(&self, particles: usize, time_step: f64, quad_order: usize) -> XResult<f64> {
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
            .map(|_| self.simulate(time_step, quad_order).unwrap())
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
    pub fn variance(&self, particles: usize, time_step: f64, quad_order: usize) -> XResult<f64> {
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
        let mean = self.mean(particles, time_step, quad_order)?;
        Ok((0..particles)
            .into_par_iter()
            .map(|_| {
                let value = self.simulate(time_step, quad_order).unwrap();
                (value - mean).powi(2)
            })
            .into_par_iter()
            .sum::<f64>()
            / particles as f64)
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
        let nodes_weights = nodes_weights_transform(0.0, duration - slag, &nodes_weights_pairs);
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
