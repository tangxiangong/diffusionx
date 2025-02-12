//! Brownian motion simulation
//!
//! This module provides functions for simulating Brownian motion.
//!

use crate::simulation::Simulation;

/// Brownian motion simulation
///
/// This struct represents a Brownian motion simulation.
///
/// # Fields
///
/// * `start_position` - The starting position of the Brownian motion.
/// * `diffusion_coefficient` - The diffusion coefficient of the Brownian motion.
pub struct BM {
    pub start_position: f64,
    pub diffusion_coefficient: f64,
}

impl BM {
    /// Create a new Brownian motion simulation
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position of the Brownian motion.
    /// * `diffusion_coefficient` - The diffusion coefficient of the Brownian motion.
    pub fn new(start_position: f64, diffusion_coefficient: f64) -> Self {
        Self {
            start_position,
            diffusion_coefficient,
        }
    }
}

/// Brownian motion simulation
///
/// This struct represents a Brownian motion simulation.
///
impl Simulation for BM {
    type Parameters = (f64, f64);
    type Results = (Vec<f64>, Vec<f64>);

    fn simulate(&self, parameters: Self::Parameters) -> Self::Results {
        let (_tau, _duration) = parameters;

        todo!()
    }
}
