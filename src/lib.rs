#[doc = include_str!("../README.md")]
/// Error handling module for the diffusionx crate
///
/// This module defines error types and result types used throughout the crate.
mod error;
pub use error::*;

/// Random number generation module
///
/// Provides implementations for various random number distributions used in stochastic processes.
pub mod random;

/// Stochastic process simulation module
///
/// Contains implementations of various stochastic processes and simulation algorithms.
pub mod simulation;

/// Utility functions and algorithms
///
/// Contains helper functions and algorithms used by other modules.
pub mod utils;

/// Visualization module for plotting simulation results
///
/// This module is only available when the "visualize" feature is enabled.
#[cfg(feature = "visualize")]
pub mod visualize;
