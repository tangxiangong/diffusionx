//! Simulation module
//!

use crate::{SimulationError, XResult};
use derive_builder::Builder;
use num_traits::Num;

pub type Pair<Time, Position> = (Vec<Time>, Vec<Position>);

/// Simulation trait
///
/// This trait represents a simulation.
///
/// # Arguments
///
/// * `Parameters` - The parameters of the simulation.
/// * `Results` - The results of the simulation.
///
/// # Returns
///
/// The results of the simulation.
pub trait Simulation {
    type Time: Num;
    type Position: Num;
    fn simulate(&self, parameters: Params) -> XResult<Pair<Self::Time, Self::Position>>;
}

#[derive(Debug, Clone, Copy, Default, Builder)]
#[builder(pattern = "mutable")]
pub struct Params {
    /// Time step for continuous-time stochastic process
    #[builder(setter(strip_option, into), default)]
    time_step: Option<f64>,
    /// Duration of the simulation for continuous-time stochastic process
    #[builder(setter(strip_option, into), default)]
    duration: Option<f64>,
}

impl Params {
    pub fn time_step(&self) -> XResult<f64> {
        if self.time_step.is_none() {
            Err(SimulationError::InvalidTimeStep("time_step is not set".to_string()).into())
        } else if self.time_step.unwrap() <= 0.0 {
            Err(SimulationError::InvalidTimeStep("time_step must be positive".to_string()).into())
        } else {
            Ok(self.time_step.unwrap())
        }
    }

    pub fn duration(&self) -> XResult<f64> {
        if self.duration.is_none() {
            Err(SimulationError::InvalidTimeInterval("duration is not set".to_string()).into())
        } else if self.duration.unwrap() <= 0.0 {
            Err(
                SimulationError::InvalidTimeInterval("duration must be positive".to_string())
                    .into(),
            )
        } else {
            Ok(self.duration.unwrap())
        }
    }
}

pub trait MomentMC : Simulation {
    fn raw_moment(&self, params: Params, order: i32, particles: usize) -> XResult<f64>;
    fn central_moment(&self, params: Params, order: i32, particles: usize) -> XResult<f64>;
}

mod bm;
pub use bm::*;

#[cfg(feature = "nightly")]
pub mod nightly;