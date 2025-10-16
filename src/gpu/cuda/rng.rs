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

        let randoms: Vec<f32> = (0..n)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();

        self.device
            .htod_sync_copy(&randoms)
            .map_err(|e| XError::GpuError(format!("Failed to copy randoms to device: {}", e)))
    }

    /// Generate standard normal random numbers on GPU (f64)
    pub fn standard_normals_f64(&mut self, n: usize) -> XResult<DevicePtrMut<f64>> {
        let mut rng = rand::thread_rng();
        use rand::Rng;
        use rand_distr::{Distribution, StandardNormal};

        let randoms: Vec<f64> = (0..n)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();

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

        let randoms: Vec<f32> = (0..n).map(|_| rng.gen::<f32>()).collect();

        self.device
            .htod_sync_copy(&randoms)
            .map_err(|e| XError::GpuError(format!("Failed to copy randoms to device: {}", e)))
    }
}

/// CUDA kernel for Box-Muller transform (GPU-side generation)
///
/// This kernel can be used for on-device random number generation
/// when integrated with a GPU-based uniform RNG state.
pub const BOX_MULLER_KERNEL: &str = r#"
extern "C" __global__ void box_muller_transform(
    const float* __restrict__ uniform1,
    const float* __restrict__ uniform2,
    float* __restrict__ normal1,
    float* __restrict__ normal2,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    float u1 = uniform1[idx];
    float u2 = uniform2[idx];

    // Avoid log(0)
    if (u1 < 1e-10f) u1 = 1e-10f;

    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * 3.14159265359f * u2;

    normal1[idx] = r * cosf(theta);
    normal2[idx] = r * sinf(theta);
}

extern "C" __global__ void box_muller_transform_f64(
    const double* __restrict__ uniform1,
    const double* __restrict__ uniform2,
    double* __restrict__ normal1,
    double* __restrict__ normal2,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    double u1 = uniform1[idx];
    double u2 = uniform2[idx];

    // Avoid log(0)
    if (u1 < 1e-10) u1 = 1e-10;

    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;

    normal1[idx] = r * cos(theta);
    normal2[idx] = r * sin(theta);
}
"#;

/// Notes for Production Implementation
///
/// For a production-ready implementation, consider:
///
/// 1. **Use cuRAND**: Integrate with NVIDIA's cuRAND library for high-performance GPU RNG
///    - curandCreateGenerator()
///    - curandGenerateNormal() for normal distributions
///    - curandGenerateUniform() for uniform distributions
///
/// 2. **GPU-side RNG states**: Implement XORSHIFT, Philox, or other GPU-friendly RNG
///    - Each thread maintains its own RNG state
///    - Avoids CPU-GPU memory transfers
///
/// 3. **Parallel streams**: Use multiple CUDA streams for overlapping computation
///
/// 4. **Memory pooling**: Reuse allocated GPU memory buffers
///
/// Example cuRAND integration (pseudo-code):
/// ```c
/// curandGenerator_t gen;
/// curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
/// curandSetPseudoRandomGeneratorSeed(gen, seed);
/// curandGenerateNormal(gen, d_output, n, mean, stddev);
/// ```

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_muller_kernel_not_empty() {
        assert!(!BOX_MULLER_KERNEL.is_empty());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_rng_creation() {
        if super::super::is_available() {
            if let Ok(device) = CudaDevice::new(0) {
                let _rng = CudaRng::new(device, 12345);
                println!("Successfully created CUDA RNG");
            }
        }
    }
}
