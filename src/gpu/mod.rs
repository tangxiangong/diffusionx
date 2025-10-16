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

pub mod montecarlo;

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

    #[test]
    fn test_backend_availability() {
        let auto = GpuBackend::Auto;
        let _is_available = auto.is_available();
        // Result depends on system
    }
}
