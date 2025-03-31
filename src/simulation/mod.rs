//! Stochastic process simulation module
//!
//! This module provides implementations of various stochastic processes and numerical methods
//! for simulating their sample paths. The implementations are optimized for performance and accuracy.
//!
//! # Continuous-time processes
//!
//! - [Brownian motion](continuous::Bm) - Standard and multidimensional Brownian motion
//! - [Lévy process](continuous::Levy) - Processes with stationary and independent increments
//! - [Subordinator](continuous::Subordinator) - Non-decreasing Lévy processes
//! - [Inverse subordinator](continuous::Subordinator) - Time-changed processes via subordinator inverses
//! - [Generalized Langevin equation](continuous::GeneralizedLangevin) - Extensions of the standard Langevin equation
//! - [Subordinated Langevin equation](continuous::SubordinatedLangevin) - Time-changed Langevin dynamics
//! - [Fractional Brownian motion](continuous::Fbm) - Long-range dependent Gaussian process
//! - [Lévy walk](continuous::LevyWalk) - Continuous-time random walk with correlated jump sizes and waiting times
//! - [Ornstein-Uhlenbeck process](continuous::OrnsteinUhlenbeck) - Mean-reverting stochastic process
//!
//! # Jump processes
//!
//! - [Poisson process](jump::Poisson) - Counting process with exponential waiting times
//! - [Continuous time random walk](jump::CTRW) - Random walk with random waiting times
//! - [Birth-death process](jump::BirthDeath) - Integer-valued Markov process modeling population dynamics
//!
//! # Discrete processes
//!
//! - [Random walk](discrete::RandomWalk) - Discrete-time random walk
//!
//! # Additional components
//!
//! - [Traits](traits) - Common traits for stochastic processes
//! - [Prelude](prelude) - Common imports for the simulation module
//! - [Functional](functional) - Functional analysis tools for stochastic processes

pub mod prelude;

pub mod traits;

pub mod continuous;

pub mod jump;

pub mod discrete;

pub mod functional;
