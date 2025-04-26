//! Stochastic process simulation module
//!
//! This module provides implementations of various stochastic processes and numerical methods
//! for simulating their sample paths. The implementations are optimized for performance and accuracy.
//!
//! # Continuous-time processes
//!
//! - Brownian motion - Standard and multidimensional Brownian motion
//! - Lévy process - Processes with stationary and independent increments
//! - Subordinator - Non-decreasing Lévy processes
//! - Inverse subordinator - Time-changed processes via subordinator inverses
//! - Generalized Langevin equation - Extensions of the standard Langevin equation
//! - Subordinated Langevin equation - Time-changed Langevin dynamics
//! - Fractional Brownian motion - Long-range dependent Gaussian process
//! - Lévy walk - Continuous-time random walk with correlated jump sizes and waiting times
//! - Ornstein-Uhlenbeck process - Mean-reverting stochastic process
//! - Brownian bridge - Brownian motion conditioned to hit origin at the end
//! - Brownian excursion  - Brownian motion conditioned to be positive and to take the value 0 at time 1
//! - Brownian meander
//!
//! # Jump processes
//!
//! - Poisson process - Counting process with exponential waiting times
//! - Continuous time random walk - Random walk with random waiting times
//! - Birth-death process - Integer-valued Markov process modeling population dynamics
//!
//! # Discrete processes
//!
//! - Random walk - Discrete-time random walk
//!

pub mod prelude;

pub mod traits;

pub mod continuous;

pub mod jump;

pub mod discrete;

pub mod functional;
