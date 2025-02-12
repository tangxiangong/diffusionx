//! Traits for simulations

/// Simulation trait
///
/// This trait represents a simulation.
///
pub trait Simulation {
    type Parameters;
    type Results;
    fn simulate(&self, parameters: Self::Parameters) -> Self::Results;
}
