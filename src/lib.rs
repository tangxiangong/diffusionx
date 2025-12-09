#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]

#[cfg_attr(feature = "mimalloc", global_allocator)]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

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
#[cfg_attr(docsrs, doc(cfg(feature = "visualize")))]
pub mod visualize;

/// GPU acceleration module
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod gpu;
