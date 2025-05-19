//! Stochastic process simulation module
//!
//! This module provides implementations of various stochastic processes and numerical methods
//! for simulating their sample paths. The implementations are optimized for performance and accuracy.
//!
//! # Continuous-time processes
//!
//! - Brownian motion [continuous::Bm]
//! - Lévy process [continuous::Levy]
//! - Cauchy process [continuous::Cauchy]
//! - Subordinator [continuous::Subordinator]
//! - Inverse subordinator [continuous::InvSubordinator]
//! - Generalized Langevin equation [continuous::GeneralizedLangevin]
//! - Subordinated Langevin equation [continuous::SubordinatedLangevin]
//! - Fractional Brownian motion [continuous::FBM]
//! - Lévy walk [continuous::LevyWalk]
//! - Ornstein-Uhlenbeck process [continuous::OrnsteinUhlenbeck]
//! - Brownian bridge [continuous::BrownianBridge]
//! - Brownian excursion [continuous::BrownianExcursion]
//! - Brownian meander [continuous::BrownianMeander]
//! - Gamma process [continuous::Gamma]
//! - Geometric Brownian motion [continuous::GeometricBm]
//!
//! # Point processes
//!
//! - Poisson process [point::Poisson]
//! - Continuous time random walk [point::CTRW]
//! - Birth-death process [point::BirthDeath]
//!
//! # Discrete processes
//!
//! - Random walk [discrete::RandomWalk]
//! - Lattice random walk [discrete::LatticeRandomWalk]
//!

pub mod prelude;

pub mod basic;

pub mod continuous;

pub mod point;

pub mod discrete;
