#[doc = include_str!("../README.md")]
mod error;
pub use error::*;
pub mod random;
pub mod simulation;
pub mod utils;
// #[cfg(feature = "visualize")]
// pub mod visualize;
