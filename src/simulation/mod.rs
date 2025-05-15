//! Stochastic process simulation module
//!
//! This module provides implementations of various stochastic processes and numerical methods
//! for simulating their sample paths. The implementations are optimized for performance and accuracy.
//!
//! # Continuous-time processes
//!
//! - Brownian motion
//! - Lévy process
//! - Cauchy process
//! - Subordinator
//! - Inverse subordinator
//! - Generalized Langevin equation
//! - Subordinated Langevin equation
//! - Fractional Brownian motion
//! - Lévy walk
//! - Ornstein-Uhlenbeck process
//! - Brownian bridge
//! - Brownian excursion
//! - Brownian meander
//! - Gamma process
//! - Geometric Brownian motion
//!
//! # Point processes
//!
//! - Poisson process
//! - Lévy walk
//! - Continuous time random walk
//! - Birth-death process
//!
//! # Discrete processes
//!
//! - Random walk
//!

pub mod prelude;

pub mod basic;

pub mod continuous;

pub mod point;

pub mod discrete;

pub mod functional;
