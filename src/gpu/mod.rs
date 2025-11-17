//! GPU acceleration module for stochastic process simulation
//!
//! This module provides GPU-accelerated implementations of stochastic processes
//! using CUDA (NVIDIA GPUs) and Metal (Apple GPUs).
//!
//! # Features
//!
//! - `cuda`: Enable CUDA support for NVIDIA GPUs
//! - `metal`: Enable Metal support for Apple GPUs
//!
//! # Example
//!
//! ```rust,ignore
//! use diffusionx::gpu::{GpuBackend, GpuSimulator};
//! use diffusionx::simulation::continuous::Bm;
//!
//! // Create GPU simulator
//! let simulator = GpuSimulator::new(GpuBackend::Cuda)?;
//!
//! // Simulate Brownian motion on GPU
//! let bm = Bm::default();
//! let trajectories = simulator.simulate_bm_cuda(&bm, 1.0, 0.01, 1000)?;
//! ```

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "metal")]
pub mod metal;

mod backend;
pub use backend::*;

mod simulator;
pub use simulator::*;

mod traits;
pub use traits::*;

pub mod mc;

mod gpu_moment;
pub use gpu_moment::*;

pub mod random;
