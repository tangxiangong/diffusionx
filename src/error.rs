//! Error handling for the diffusionx crate
//!
//! This module provides error types and result types used throughout the crate.
//! It defines various error categories for different components such as random number
//! sampling, simulation processes, and visualization.

use rand::distr::uniform::Error as UniformError;
use rand_distr::{ExpError, NormalError, PoissonError};
use thiserror::Error;

/// Result type for the diffusionx crate
///
/// This is a specialized Result type that uses XError as the error type.
pub type XResult<T> = Result<T, XError>;

/// Main error type for the diffusionx crate
///
/// This enum represents all possible errors that can occur within the crate.
/// It handles errors from various sources, including random number generation,
/// simulation processes, and visualization.
#[derive(Error, Debug, PartialEq, Clone)]
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
    /// Error for visualization
    #[error("Visualization Error: {0}")]
    VisualizationError(#[from] PlotterError),
    /// Error for invalid parameters in various contexts
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    /// Error for non-positive definite matrix
    #[error("Circulant embedding matrix is not positive definite, eigenvalue: {0}")]
    NotPositiveDefinite(f64),
}

/// Error type for stable distribution sampling
///
/// This enum represents the errors that can occur when sampling from
/// stable distributions with invalid parameters.
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

/// Error type for simulation processes
///
/// This enum represents errors that can occur during
/// the simulation of stochastic processes.
#[derive(Error, Debug, PartialEq, Eq, Clone)]
pub enum SimulationError {
    /// Error for invalid parameters in simulation configuration
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    /// Error for invalid time step in simulation
    #[error("Invalid time step: {0}")]
    InvalidTimeStep(String),
    /// Error for invalid time interval in simulation
    #[error("Invalid time interval: {0}")]
    InvalidTimeInterval(String),
    /// Error for unknown simulation errors
    #[error("Unknown error, simulation failed")]
    Unknown,
}

/// Error wrapper for the `plotters` crate visualization errors
///
/// This enum provides a more specific error classification for
/// plotting and visualization errors.
#[derive(Error, Debug, PartialEq, Eq, Clone)]
pub enum PlotterError {
    /// Error for invalid visualization configuration
    #[error("Config Error: {0}")]
    ConfigError(String),
    /// Error for invalid color specification
    #[error("Invalid color: {0}")]
    InvalidColor(String),
    /// Error from Plotters DrawingAreaErrorKind
    #[error("Plotter Error: {0}")]
    DrawingError(String),
}

/// Implementation of From trait to convert Plotters errors to XError
///
/// This allows for seamless error propagation from the plotting library.
impl<E: std::error::Error + Send + Sync> From<plotters::drawing::DrawingAreaErrorKind<E>>
    for XError
{
    fn from(err: plotters::drawing::DrawingAreaErrorKind<E>) -> Self {
        PlotterError::DrawingError(err.to_string()).into()
    }
}
