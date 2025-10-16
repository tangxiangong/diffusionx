//! Metal random number generation for GPU-accelerated stochastic processes
//!
//! This module provides GPU-based random number generation for Metal backend.

use crate::{XError, XResult};

#[cfg(feature = "metal")]
use metal::*;

/// Random number generator for Metal
///
/// This generates random numbers on CPU and transfers to GPU.
/// For production use, consider implementing GPU-side RNG for better performance.
#[cfg(feature = "metal")]
pub struct MetalRng {
    device: Device,
}

#[cfg(feature = "metal")]
impl MetalRng {
    /// Create a new Metal RNG
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// Generate standard normal random numbers
    ///
    /// # Arguments
    ///
    /// * `n` - Number of random numbers to generate
    ///
    /// # Returns
    ///
    /// Buffer containing the generated random numbers
    pub fn standard_normals_f32(&self, n: usize) -> XResult<Buffer> {
        use rand_distr::{Distribution, StandardNormal};

        let mut rng = rand::rng();
        let randoms: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();

        let buffer = self.device.new_buffer_with_data(
            randoms.as_ptr() as *const _,
            (randoms.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(buffer)
    }

    /// Generate normal random numbers with given mean and standard deviation
    pub fn normals_f32(&self, mean: f32, std_dev: f32, n: usize) -> XResult<Buffer> {
        use rand_distr::{Distribution, Normal};

        let normal = Normal::new(mean, std_dev)
            .map_err(|e| XError::InvalidParameters(format!("Invalid normal parameters: {}", e)))?;

        let mut rng = rand::rng();
        let randoms: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng)).collect();

        let buffer = self.device.new_buffer_with_data(
            randoms.as_ptr() as *const _,
            (randoms.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(buffer)
    }

    /// Generate uniform random numbers in [0, 1)
    pub fn uniform_f32(&self, n: usize) -> XResult<Buffer> {
        use rand::Rng;

        let mut rng = rand::rng();
        let randoms: Vec<f32> = (0..n).map(|_| rng.random::<f32>()).collect();

        let buffer = self.device.new_buffer_with_data(
            randoms.as_ptr() as *const _,
            (randoms.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(buffer)
    }

    /// Generate two sets of uniform random numbers for Box-Muller transform
    pub fn uniform_pairs_f32(&self, n: usize) -> XResult<(Buffer, Buffer)> {
        use rand::Rng;

        let mut rng = rand::rng();
        let uniform1: Vec<f32> = (0..n).map(|_| rng.random::<f32>()).collect();
        let uniform2: Vec<f32> = (0..n).map(|_| rng.random::<f32>()).collect();

        let buffer1 = self.device.new_buffer_with_data(
            uniform1.as_ptr() as *const _,
            (uniform1.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let buffer2 = self.device.new_buffer_with_data(
            uniform2.as_ptr() as *const _,
            (uniform2.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok((buffer1, buffer2))
    }

    /// Transform uniform random numbers to normal using Box-Muller on GPU
    pub fn box_muller_transform(
        &self,
        library: &Library,
        uniform1: &Buffer,
        uniform2: &Buffer,
        n: usize,
    ) -> XResult<(Buffer, Buffer)> {
        let kernel = library
            .get_function("box_muller_transform", None)
            .map_err(|e| XError::GpuError(format!("Failed to get kernel function: {}", e)))?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| XError::GpuError(format!("Failed to create compute pipeline: {}", e)))?;

        // Create output buffers
        let normal1 = self.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let normal2 = self.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer and encoder
        let command_queue = self.device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(uniform1), 0);
        encoder.set_buffer(1, Some(uniform2), 0);
        encoder.set_buffer(2, Some(&normal1), 0);
        encoder.set_buffer(3, Some(&normal2), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<i32>() as u64,
            &(n as i32) as *const i32 as *const _,
        );

        // Calculate thread configuration
        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new(((n + 255) / 256) as u64, 1, 1);

        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((normal1, normal2))
    }
}

/// Notes for Production Implementation
///
/// For a production-ready GPU RNG implementation, consider:
///
/// 1. **GPU-side RNG**: Implement Philox, Threefry, or PCG on GPU
///    - Each thread maintains its own state
///    - Avoids CPU-GPU memory transfers
///    - Much faster for large-scale simulations
///
/// 2. **Memory pooling**: Reuse allocated buffers
///    - Pre-allocate buffers for common sizes
///    - Reduce allocation overhead
///
/// 3. **Async generation**: Generate random numbers asynchronously
///    - Use Metal's command buffer callbacks
///    - Overlap generation with computation
///
/// 4. **Quality considerations**:
///    - Ensure statistical quality of GPU RNG
///    - Test for correlations between threads
///    - Use appropriate seeding strategies
///
/// Example Philox RNG implementation (pseudo-code):
/// ```metal
/// struct PhiloxState {
///     uint4 counter;
///     uint2 key;
/// };
///
/// uint4 philox4x32(PhiloxState state) {
///     // Philox algorithm implementation
///     // Returns 4 random uint32 values
/// }
/// ```

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "metal")]
    fn test_metal_rng_creation() {
        if super::super::is_available() {
            if let Some(device) = Device::system_default() {
                let _rng = MetalRng::new(device);
                println!("Successfully created Metal RNG");
            }
        }
    }

    #[test]
    #[cfg(feature = "metal")]
    fn test_generate_normals() {
        if super::super::is_available() {
            if let Some(device) = Device::system_default() {
                let rng = MetalRng::new(device);
                match rng.standard_normals_f32(100) {
                    Ok(buffer) => {
                        println!("Generated buffer of size: {} bytes", buffer.length());
                        assert_eq!(buffer.length(), (100 * std::mem::size_of::<f32>()) as u64);
                    }
                    Err(e) => println!("Error generating normals: {}", e),
                }
            }
        }
    }
}
