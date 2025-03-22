//! Utility functions and algorithms module
//!
//! This module provides various utility functions and algorithms used throughout the crate,
//! including mathematical functions, data structures, and computational methods specific
//! to stochastic process simulations.
//!
//! # Components
//!
//! - [Auxiliary functions](functions) - Mathematical and computational helper functions
//!   used in various simulation algorithms
//! - [Circulant embedding](circulant_embedding) - Fast and efficient implementation of the circulant embedding
//!   method for simulating stationary Gaussian processes with given covariance structure

mod functions;
pub use functions::*;

mod circulant_embedding;
pub use circulant_embedding::*;
