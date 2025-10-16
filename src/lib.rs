#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
//!
//! ## 简体中文
//!
//! 中文版本可见[这里](https://github.com/tangxiangong/diffusionx/blob/stable/README-zh.md).

mod error;
pub use error::*;

/// Random number generation module
pub mod random;

/// Stochastic process simulation module
pub mod simulation;

/// Utility functions and algorithms
pub mod utils;

/// Visualization module
#[cfg(feature = "visualize")]
pub mod visualize;

/// GPU acceleration module
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod gpu;
