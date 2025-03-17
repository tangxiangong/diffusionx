//! Visualization utilities for the simulation.
//!
//! This module visualizing the simulation by using the crate `[plotters](https://crates.io/crates/plotters)`.
//!
//! # Dependencies
//!
//! ## Ubuntu Linux
//!
//! ```bash
//! sudo apt install pkg-config libfreetype6-dev libfontconfig1-dev
//! ```
//!

pub mod config;
pub use config::*;

pub mod draw;
pub use draw::*;
