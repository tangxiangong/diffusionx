//! Random number generation module
//!
//! This module provides functions and types for generating random numbers from various probability distributions.
//! These random number generators are optimized for use in stochastic process simulations.
//!
//! # Available distributions
//!
//! - Exponential distribution
//! - Normal (Gaussian) distribution
//! - Poisson distribution
//! - Stable distribution
//! - Uniform distribution
//! - Gamma distribution

pub mod exponential;
pub mod gamma;
pub mod normal;
pub mod poisson;
pub mod stable;
pub mod uniform;
