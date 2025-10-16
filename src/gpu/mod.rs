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
//! let (times, positions) = simulator.simulate(&bm, 1.0, 0.01, 1000)?;
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

use crate::XResult;

/// GPU backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    #[cfg(feature = "cuda")]
    /// CUDA backend for NVIDIA GPUs
    Cuda,

    #[cfg(feature = "metal")]
    /// Metal backend for Apple GPUs
    Metal,

    /// Auto-select the best available backend
    Auto,
}

impl GpuBackend {
    /// Detect and return the best available GPU backend
    pub fn detect() -> XResult<Self> {
        #[cfg(feature = "cuda")]
        {
            if cuda::is_available() {
                return Ok(Self::Cuda);
            }
        }

        #[cfg(feature = "metal")]
        {
            if metal::is_available() {
                return Ok(Self::Metal);
            }
        }

        Err(crate::XError::GpuError(
            "No GPU backend available. Please enable 'cuda' or 'metal' feature.".to_string(),
        ))
    }

    /// Check if this backend is available on the current system
    pub fn is_available(&self) -> bool {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda => cuda::is_available(),

            #[cfg(feature = "metal")]
            Self::Metal => metal::is_available(),

            Self::Auto => {
                #[cfg(feature = "cuda")]
                if cuda::is_available() {
                    return true;
                }

                #[cfg(feature = "metal")]
                if metal::is_available() {
                    return true;
                }

                false
            }
        }
    }
}

impl Default for GpuBackend {
    fn default() -> Self {
        Self::Auto
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_detection() {
        match GpuBackend::detect() {
            Ok(backend) => {
                println!("Detected GPU backend: {:?}", backend);
                assert!(backend.is_available());
            }
            Err(_) => {
                println!("No GPU backend available");
            }
        }
    }
}
