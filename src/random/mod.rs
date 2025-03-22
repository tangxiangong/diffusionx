//! Random number generation module
//!
//! This module provides functions and types for generating random numbers from various probability distributions.
//! These random number generators are optimized for use in stochastic process simulations.
//!
//! The implementations are designed to be used in parallel environments and provide consistent,
//! high-quality random numbers with appropriate statistical properties.
//!
//! # Available distributions
//!
//! - [Exponential distribution](exponential) - For generating waiting times and decay processes
//! - [Normal (Gaussian) distribution](normal) - For Brownian motion and many other stochastic processes
//! - [Poisson distribution](poisson) - For jump processes and counting events
//! - [Stable distribution](stable) - For heavy-tailed distributions and Lévy processes
//! - [Uniform distribution](uniform) - For general random sampling and basis of other distributions

pub mod exponential;
pub mod normal;
pub mod poisson;
pub mod stable;
pub mod uniform;
