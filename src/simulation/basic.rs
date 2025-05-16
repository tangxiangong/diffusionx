//! Traits and structs for stochastic processes
//!
//! ## Traits
//! - Continuous process [ContinuousProcess]
//! - Point process [PointProcess]
//! - Discrete process [DiscreteProcess]
//! - Moment [Moment]
//! - Inverse process [Inverse]
//!
//! ## Structs
//! - ContinuousTrajectory [ContinuousTrajectory]
//! - DiscreteTrajectory [DiscreteTrajectory]
//! - PointTrajectory [PointTrajectory]
//! - TAMSD [TAMSD]
//!

pub type Pair = (Vec<f64>, Vec<f64>);
pub type DiscretePair = (Vec<usize>, Vec<f64>);

mod continuous;
pub use continuous::*;

mod discrete;
pub use discrete::*;

mod point;
pub use point::*;

mod moment;
pub use moment::*;

mod tamsd;
pub use tamsd::*;

mod inverse;
pub use inverse::*;

mod functional;
pub use functional::*;
