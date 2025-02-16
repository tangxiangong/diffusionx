//! Simulation module
//!

use crate::XResult;
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
    type Params;
    fn simulate(&self, params: Self::Params) -> XResult<Pair<Self::Time, Self::Position>>;
}

pub trait Moment: Simulation {
    fn raw_moment(&self, params: Self::Params, order: i32, particles: usize) -> XResult<f64>;
    fn central_moment(&self, params: Self::Params, order: i32, particles: usize) -> XResult<f64>;
}

mod bm;
pub use bm::*;

#[cfg(feature = "nightly")]
pub mod nightly;
