//! Visualization module for stochastic process simulations
//!
//! This module provides tools and utilities for visualizing simulation results
//! from stochastic processes. It uses the [`plotters`](https://crates.io/crates/plotters)
//! crate as the underlying plotting library.
//!
//! The module allows for creating publication-quality plots of time series,
//! histograms, and other visualizations of stochastic processes with customizable
//! styling options.
//!
//! # System Dependencies
//!
//! The visualization module requires certain system libraries for font rendering:
//!
//! ## Ubuntu Linux
//!
//! ```bash
//! sudo apt install pkg-config libfreetype6-dev libfontconfig1-dev
//! ```

pub mod config;
pub use config::*;

pub mod draw;
pub use draw::*;
