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
    #[error("Sample Uniform Distribution Error: {0}")]
    UniformSampleError(#[from] UniformError),
    #[error("Sample Normal Distribution Error: {0}")]
    NormalSampleError(#[from] NormalError),
    #[error("Sample Poisson Distribution Error: {0}")]
    PoissonSampleError(#[from] PoissonError),
    #[error("Sample Exponential Distribution Error: {0}")]
    ExpSampleError(#[from] ExpError),
    #[error("Sample Stable Distribution Error: {0}")]
    StableSampleError(#[from] StableError),
    #[error("Probability must be between 0 and 1")]
    BoolSampleError,
    #[error("Simulate Error: {0}")]
    SimulateError(#[from] SimulationError),
}

#[derive(Error, Debug, PartialEq, Eq, Clone, Copy)]
pub enum StableError {
    #[error("Index of stability must be in the range (0, 2]")]
    InvalidIndex,
    #[error("Skewness parameter must be in the range [-1, 1]")]
    InvalidSkewness,
    #[error("Scale parameter must be positive")]
    InvalidScale,
    #[error("Location parameter must be a real number")]
    InvalidLocation,
    #[error("Index of stability must be in the range (0, 1)")]
    InvalidSkewIndex,
}

#[derive(Error, Debug, PartialEq, Eq, Clone)]
pub enum SimulationError {
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Invalid time step: {0}")]
    InvalidTimeStep(String),
    #[error("Invalid time interval: {0}")]
    InvalidTimeInterval(String),
    #[error("Unknown error, simulation failed")]
    Unknown,
}
