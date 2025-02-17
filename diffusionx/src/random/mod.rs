//! Random number generation
//!
//! This module provides functions for generating random numbers.
//!
//! The functions in this module are designed to be used in parallel environments.
//!
//! Supported distributions:
//! - Exponential distribution in [exponential]
//! - Normal distribution in [normal]
//! - Poisson distribution in [poisson]
//! - Stable distribution in [stable]
//! - Uniform distribution in [uniform]

pub mod exponential;
pub mod normal;
pub mod poisson;
pub mod stable;
pub mod uniform;
