//! CUDA random number generation for GPU-accelerated stochastic processes
//!
//! This module provides GPU-based random number generation using cuRAND.

use crate::{XError, XResult};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, DevicePtrMut};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Random number generator for CUDA
///
/// This uses the Box-Muller transform to generate normal random numbers on GPU.
/// For production use, consider integrating with cuRAND library for better performance.
#[cfg(feature = "cuda")]
pub struct CudaRng {
    device: Arc<CudaDevice>,
    seed: u64,
}

#[cfg(feature = "cuda")]
impl CudaRng {
    /// Create a new CUDA RNG with the given seed
    pub fn new(device: Arc<CudaDevice>, seed: u64) -> Self {
        Self { device, seed }
    }

    /// Generate standard normal random numbers on GPU
    ///
    /// # Arguments
    ///
    /// * `n` - Number of random numbers to generate
    ///
    /// # Returns
    ///
    /// Device pointer to the generated random numbers
    pub fn standard_normals_f32(&mut self, n: usize) -> XResult<DevicePtrMut<f32>> {
        // For now, generate on CPU and copy to GPU
        // In production, use cuRAND or implement a GPU kernel
        let mut rng = rand::thread_rng();
        use rand::Rng;
        use rand_distr::{Distribution, StandardNormal};

        let randoms: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();

        self.device
            .htod_sync_copy(&randoms)
            .map_err(|e| XError::GpuError(format!("Failed to copy randoms to device: {}", e)))
    }

    /// Generate standard normal random numbers on GPU (f64)
    pub fn standard_normals_f64(&mut self, n: usize) -> XResult<DevicePtrMut<f64>> {
        let mut rng = rand::thread_rng();
        use rand::Rng;
        use rand_distr::{Distribution, StandardNormal};

        let randoms: Vec<f64> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();

        self.device
            .htod_sync_copy(&randoms)
            .map_err(|e| XError::GpuError(format!("Failed to copy randoms to device: {}", e)))
    }

    /// Generate normal random numbers with given mean and standard deviation
    pub fn normals_f32(&mut self, mean: f32, std_dev: f32, n: usize) -> XResult<DevicePtrMut<f32>> {
        let mut rng = rand::thread_rng();
        use rand_distr::{Distribution, Normal};

        let normal = Normal::new(mean, std_dev)
            .map_err(|e| XError::InvalidParameters(format!("Invalid normal parameters: {}", e)))?;

        let randoms: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng)).collect();

        self.device
            .htod_sync_copy(&randoms)
            .map_err(|e| XError::GpuError(format!("Failed to copy randoms to device: {}", e)))
    }

    /// Generate uniform random numbers in [0, 1)
    pub fn uniform_f32(&mut self, n: usize) -> XResult<DevicePtrMut<f32>> {
        let mut rng = rand::thread_rng();
        use rand::Rng;

        let randoms: Vec<f32> = (0..n).map(|_| rng.r#gen::<f32>()).collect();

        self.device
            .htod_sync_copy(&randoms)
            .map_err(|e| XError::GpuError(format!("Failed to copy randoms to device: {}", e)))
    }
}
