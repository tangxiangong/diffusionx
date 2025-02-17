//! Error module
//!
//! This module provides error types.

use rand::distr::uniform::Error as UniformError;
use rand_distr::{ExpError, NormalError, PoissonError};
use thiserror::Error;

/// Result type for the crate
pub type XResult<T> = Result<T, XError>;

/// Error type for the crate
///
/// This enum represents the errors that can occur in the crate.
///
#[derive(Error, Debug, PartialEq, Eq, Clone)]
pub enum XError {
    /// Error for sampling from the uniform distribution
    #[error("Sample Uniform Distribution Error: {0}")]
    UniformSampleError(#[from] UniformError),
    /// Error for sampling from the normal distribution
    #[error("Sample Normal Distribution Error: {0}")]
    NormalSampleError(#[from] NormalError),
    /// Error for sampling from the Poisson distribution
    #[error("Sample Poisson Distribution Error: {0}")]
    PoissonSampleError(#[from] PoissonError),
    /// Error for sampling from the exponential distribution
    #[error("Sample Exponential Distribution Error: {0}")]
    ExpSampleError(#[from] ExpError),
    /// Error for sampling from the stable distribution
    #[error("Sample Stable Distribution Error: {0}")]
    StableSampleError(#[from] StableError),
    /// Error for sampling from the boolean distribution
    #[error("Probability must be between 0 and 1")]
    BoolSampleError,
    /// Error for simulating the process
    #[error("Simulate Error: {0}")]
    SimulateError(#[from] SimulationError),
}

/// Error for sampling from the stable distribution
#[derive(Error, Debug, PartialEq, Eq, Clone, Copy)]
pub enum StableError {
    /// Error for invalid index of stability
    #[error("Index of stability must be in the range (0, 2]")]
    InvalidIndex,
    /// Error for invalid skewness parameter
    #[error("Skewness parameter must be in the range [-1, 1]")]
    InvalidSkewness,
    /// Error for invalid scale parameter
    #[error("Scale parameter must be positive")]
    InvalidScale,
    /// Error for invalid location parameter
    #[error("Location parameter must be a real number")]
    InvalidLocation,
    /// Error for invalid index of skewness
    #[error("Index of skewness must be in the range (0, 1)")]
    InvalidSkewIndex,
}

/// Error for simulating the process
#[derive(Error, Debug, PartialEq, Eq, Clone)]
pub enum SimulationError {
    /// Error for invalid parameters
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    /// Error for invalid time step
    #[error("Invalid time step: {0}")]
    InvalidTimeStep(String),
    /// Error for invalid time interval
    #[error("Invalid time interval: {0}")]
    InvalidTimeInterval(String),
    /// Error for unknown error
    #[error("Unknown error, simulation failed")]
    Unknown,
}
