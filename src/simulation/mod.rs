//! Stochastic process simulation module
//!
//! This module provides implementations of various stochastic processes and numerical methods
//! for simulating their sample paths. The implementations are optimized for performance and accuracy.

/// Common traits and process types for convenient glob imports.
///
/// Importing this module is the shortest way to bring simulation traits such as
/// `ContinuousProcess`, `PointProcess`, `DiscreteProcess`, and moment utilities
/// into scope.
pub mod prelude;

pub mod basic;

pub mod continuous;

pub mod point;

pub mod discrete;

/// Macros for generating stochastic processes
pub mod macros;
