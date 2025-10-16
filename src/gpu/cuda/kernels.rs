//! CUDA kernel implementations for stochastic process simulation
//!
//! This module contains CUDA kernel code and Rust wrappers for GPU-accelerated
//! simulation of stochastic processes.

use crate::{XError, XResult};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaFunction, DevicePtrMut, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx;
#[cfg(feature = "cuda")]
use std::path::Path;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// CUDA kernel source code for Brownian motion simulation
pub const BROWNIAN_MOTION_KERNEL: &str = r#"
extern "C" __global__ void simulate_brownian_motion(
    const float* __restrict__ random_normals,
    float* __restrict__ positions,
    float start_position,
    float diffusion_coefficient,
    float time_step,
    int num_steps,
    int num_particles
) {
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (particle_idx >= num_particles) return;

    float position = start_position;
    int offset = particle_idx * num_steps;

    // Store initial position
    positions[offset] = position;

    // Simulate trajectory
    float noise_scale = sqrtf(2.0f * diffusion_coefficient * time_step);
    for (int step = 0; step < num_steps; step++) {
        position += noise_scale * random_normals[offset + step];
        positions[offset + step + 1] = position;
    }
}

extern "C" __global__ void simulate_brownian_motion_f64(
    const double* __restrict__ random_normals,
    double* __restrict__ positions,
    double start_position,
    double diffusion_coefficient,
    double time_step,
    int num_steps,
    int num_particles
) {
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (particle_idx >= num_particles) return;

    double position = start_position;
    int offset = particle_idx * num_steps;

    // Store initial position
    positions[offset] = position;

    // Simulate trajectory
    double noise_scale = sqrt(2.0 * diffusion_coefficient * time_step);
    for (int step = 0; step < num_steps; step++) {
        position += noise_scale * random_normals[offset + step];
        positions[offset + step + 1] = position;
    }
}
"#;

/// CUDA kernel for computing statistical moments
pub const MOMENTS_KERNEL: &str = r#"
extern "C" __global__ void compute_raw_moment(
    const float* __restrict__ positions,
    float* __restrict__ moments,
    int order,
    int num_particles,
    int num_steps
) {
    int step_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (step_idx >= num_steps) return;

    float sum = 0.0f;
    for (int i = 0; i < num_particles; i++) {
        float pos = positions[i * (num_steps + 1) + step_idx];
        float value = pos;
        for (int j = 1; j < order; j++) {
            value *= pos;
        }
        sum += value;
    }

    moments[step_idx] = sum / num_particles;
}

extern "C" __global__ void compute_central_moment(
    const float* __restrict__ positions,
    const float* __restrict__ means,
    float* __restrict__ moments,
    int order,
    int num_particles,
    int num_steps
) {
    int step_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (step_idx >= num_steps) return;

    float mean = means[step_idx];
    float sum = 0.0f;

    for (int i = 0; i < num_particles; i++) {
        float pos = positions[i * (num_steps + 1) + step_idx];
        float deviation = pos - mean;
        float value = deviation;
        for (int j = 1; j < order; j++) {
            value *= deviation;
        }
        sum += value;
    }

    moments[step_idx] = sum / num_particles;
}
"#;

/// CUDA kernel for Ornstein-Uhlenbeck process
pub const OU_PROCESS_KERNEL: &str = r#"
extern "C" __global__ void simulate_ou_process(
    const float* __restrict__ random_normals,
    float* __restrict__ positions,
    float start_position,
    float theta,
    float mu,
    float sigma,
    float time_step,
    int num_steps,
    int num_particles
) {
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (particle_idx >= num_particles) return;

    float position = start_position;
    int offset = particle_idx * num_steps;

    positions[offset] = position;

    float sqrt_dt = sqrtf(time_step);
    for (int step = 0; step < num_steps; step++) {
        float drift = theta * (mu - position) * time_step;
        float diffusion = sigma * sqrt_dt * random_normals[offset + step];
        position += drift + diffusion;
        positions[offset + step + 1] = position;
    }
}
"#;

/// CUDA kernel for Geometric Brownian Motion
pub const GBM_KERNEL: &str = r#"
extern "C" __global__ void simulate_gbm(
    const float* __restrict__ random_normals,
    float* __restrict__ positions,
    float start_position,
    float mu,
    float sigma,
    float time_step,
    int num_steps,
    int num_particles
) {
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (particle_idx >= num_particles) return;

    float position = start_position;
    int offset = particle_idx * num_steps;

    positions[offset] = position;

    float sqrt_dt = sqrtf(time_step);
    for (int step = 0; step < num_steps; step++) {
        float drift = (mu - 0.5f * sigma * sigma) * time_step;
        float diffusion = sigma * sqrt_dt * random_normals[offset + step];
        position *= expf(drift + diffusion);
        positions[offset + step + 1] = position;
    }
}
"#;

/// Kernel manager for loading and caching CUDA kernels
#[cfg(feature = "cuda")]
pub struct KernelManager {
    device: Arc<CudaDevice>,
    brownian_motion_f32: Option<Arc<CudaFunction>>,
    brownian_motion_f64: Option<Arc<CudaFunction>>,
    init_curand: Option<Arc<CudaFunction>>,
    moments: Option<Arc<CudaFunction>>,
    ou_process: Option<Arc<CudaFunction>>,
    gbm: Option<Arc<CudaFunction>>,
}

/// Launch parameters for kernel execution
#[cfg(feature = "cuda")]
pub struct KernelLaunchParams {
    pub num_particles: usize,
    pub num_steps: usize,
    pub threads_per_block: usize,
}

#[cfg(feature = "cuda")]
impl KernelLaunchParams {
    pub fn new(num_particles: usize, num_steps: usize) -> Self {
        Self {
            num_particles,
            num_steps,
            threads_per_block: 256,
        }
    }

    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads_per_block = threads;
        self
    }

    pub fn get_launch_config(&self) -> LaunchConfig {
        let num_blocks = (self.num_particles + self.threads_per_block - 1) / self.threads_per_block;
        LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (self.threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

#[cfg(feature = "cuda")]
impl KernelManager {
    /// Create a new kernel manager
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            brownian_motion_f32: None,
            brownian_motion_f64: None,
            init_curand: None,
            moments: None,
            ou_process: None,
            gbm: None,
        }
    }

    /// Load kernel from .cu file if available, otherwise use embedded source
    fn load_kernel_from_file_or_source(
        &self,
        file_path: &str,
        embedded_source: &str,
        module_name: &str,
        function_names: &[&str],
    ) -> XResult<()> {
        let ptx = if Path::new(file_path).exists() {
            // Try to compile from file
            let source = std::fs::read_to_string(file_path)
                .map_err(|e| XError::GpuError(format!("Failed to read kernel file: {}", e)))?;
            compile_ptx(source)
                .map_err(|e| XError::GpuError(format!("Failed to compile PTX from file: {}", e)))?
        } else {
            // Use embedded source
            compile_ptx(embedded_source)
                .map_err(|e| XError::GpuError(format!("Failed to compile embedded PTX: {}", e)))?
        };

        self.device
            .load_ptx(ptx, module_name, function_names)
            .map_err(|e| XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        Ok(())
    }

    /// Initialize cuRAND states on device
    pub fn init_curand_states(
        &mut self,
        num_particles: usize,
        seed: u64,
    ) -> XResult<DevicePtrMut<u8>> {
        // Load init kernel if not already loaded
        if self.init_curand.is_none() {
            self.load_kernel_from_file_or_source(
                "kernels/cuda/bm.cu",
                BROWNIAN_MOTION_KERNEL,
                "curand_init_module",
                &["init_curand_states"],
            )?;

            let kernel = self
                .device
                .get_func("curand_init_module", "init_curand_states")
                .map_err(|e| XError::GpuError(format!("Failed to get init kernel: {}", e)))?;

            self.init_curand = Some(kernel);
        }

        // Allocate memory for cuRAND states (48 bytes per state for curandState)
        let state_size = 48;
        let total_size = num_particles * state_size;
        let mut d_states = self
            .device
            .alloc_zeros::<u8>(total_size)
            .map_err(|e| XError::GpuError(format!("Failed to allocate cuRAND states: {}", e)))?;

        // Launch initialization kernel
        let config = KernelLaunchParams::new(num_particles, 0).get_launch_config();
        let kernel = self.init_curand.as_ref().unwrap();

        unsafe {
            kernel
                .clone()
                .launch(config, (&d_states, seed, num_particles as i32))
                .map_err(|e| XError::GpuError(format!("Failed to launch cuRAND init: {}", e)))?;
        }

        self.device.synchronize().map_err(|e| {
            XError::GpuError(format!("Failed to synchronize after cuRAND init: {}", e))
        })?;

        Ok(d_states)
    }

    /// Load Brownian motion kernel (f32)
    pub fn load_brownian_motion_f32(&mut self) -> XResult<Arc<CudaFunction>> {
        if let Some(ref kernel) = self.brownian_motion_f32 {
            return Ok(kernel.clone());
        }

        self.load_kernel_from_file_or_source(
            "kernels/cuda/bm.cu",
            BROWNIAN_MOTION_KERNEL,
            "bm_f32",
            &["simulate_bm_f32"],
        )?;

        let kernel = self
            .device
            .get_func("bm_f32", "simulate_bm_f32")
            .map_err(|e| XError::GpuError(format!("Failed to get kernel function: {}", e)))?;

        self.brownian_motion_f32 = Some(kernel.clone());
        Ok(kernel)
    }

    /// Simulate Brownian motion on GPU (f32)
    pub fn simulate_bm_f32(
        &mut self,
        params: &KernelLaunchParams,
        start_position: f32,
        diffusion_coefficient: f32,
        time_step: f32,
        seed: u64,
    ) -> XResult<Vec<f32>> {
        // Initialize cuRAND states
        let d_states = self.init_curand_states(params.num_particles, seed)?;

        // Allocate output buffer
        let output_size = params.num_particles * (params.num_steps + 1);
        let mut d_positions = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| XError::GpuError(format!("Failed to allocate positions: {}", e)))?;

        // Load and launch kernel
        let kernel = self.load_brownian_motion_f32()?;
        let config = params.get_launch_config();

        unsafe {
            kernel
                .launch(
                    config,
                    (
                        &d_states,
                        &mut d_positions,
                        start_position,
                        diffusion_coefficient,
                        time_step,
                        params.num_steps as i32,
                        params.num_particles as i32,
                    ),
                )
                .map_err(|e| XError::GpuError(format!("Failed to launch BM kernel: {}", e)))?;
        }

        // Synchronize and copy results back
        self.device
            .synchronize()
            .map_err(|e| XError::GpuError(format!("Failed to synchronize: {}", e)))?;

        let positions = self
            .device
            .dtoh_sync_copy(&d_positions)
            .map_err(|e| XError::GpuError(format!("Failed to copy results: {}", e)))?;

        Ok(positions)
    }

    /// Load Brownian motion kernel (f64)
    pub fn load_brownian_motion_f64(&mut self) -> XResult<Arc<CudaFunction>> {
        if let Some(ref kernel) = self.brownian_motion_f64 {
            return Ok(kernel.clone());
        }

        self.load_kernel_from_file_or_source(
            "kernels/cuda/bm.cu",
            BROWNIAN_MOTION_KERNEL,
            "bm_f64",
            &["simulate_bm_f64"],
        )?;

        let kernel = self
            .device
            .get_func("bm_f64", "simulate_bm_f64")
            .map_err(|e| XError::GpuError(format!("Failed to get kernel function: {}", e)))?;

        self.brownian_motion_f64 = Some(kernel.clone());
        Ok(kernel)
    }

    /// Simulate Brownian motion on GPU (f64)
    pub fn simulate_bm_f64(
        &mut self,
        params: &KernelLaunchParams,
        start_position: f64,
        diffusion_coefficient: f64,
        time_step: f64,
        seed: u64,
    ) -> XResult<Vec<f64>> {
        // Initialize cuRAND states
        let d_states = self.init_curand_states(params.num_particles, seed)?;

        // Allocate output buffer
        let output_size = params.num_particles * (params.num_steps + 1);
        let mut d_positions = self
            .device
            .alloc_zeros::<f64>(output_size)
            .map_err(|e| XError::GpuError(format!("Failed to allocate positions: {}", e)))?;

        // Load and launch kernel
        let kernel = self.load_brownian_motion_f64()?;
        let config = params.get_launch_config();

        unsafe {
            kernel
                .launch(
                    config,
                    (
                        &d_states,
                        &mut d_positions,
                        start_position,
                        diffusion_coefficient,
                        time_step,
                        params.num_steps as i32,
                        params.num_particles as i32,
                    ),
                )
                .map_err(|e| XError::GpuError(format!("Failed to launch BM kernel: {}", e)))?;
        }

        // Synchronize and copy results back
        self.device
            .synchronize()
            .map_err(|e| XError::GpuError(format!("Failed to synchronize: {}", e)))?;

        let positions = self
            .device
            .dtoh_sync_copy(&d_positions)
            .map_err(|e| XError::GpuError(format!("Failed to copy results: {}", e)))?;

        Ok(positions)
    }

    /// Load moments kernel
    pub fn load_moments_kernel(&mut self) -> XResult<Arc<CudaFunction>> {
        if let Some(ref kernel) = self.moments {
            return Ok(kernel.clone());
        }

        let ptx = compile_ptx(MOMENTS_KERNEL)
            .map_err(|e| XError::GpuError(format!("Failed to compile PTX: {}", e)))?;

        self.device
            .load_ptx(
                ptx,
                "moments",
                &["compute_raw_moment", "compute_central_moment"],
            )
            .map_err(|e| XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        let kernel = self
            .device
            .get_func("moments", "compute_raw_moment")
            .map_err(|e| XError::GpuError(format!("Failed to get kernel function: {}", e)))?;

        self.moments = Some(kernel.clone());
        Ok(kernel)
    }

    /// Compute raw moments on GPU
    pub fn compute_raw_moments(
        &mut self,
        positions: &[f32],
        order: i32,
        num_particles: usize,
        num_steps: usize,
    ) -> XResult<Vec<f32>> {
        // Copy positions to device
        let d_positions = self
            .device
            .htod_sync_copy(positions)
            .map_err(|e| XError::GpuError(format!("Failed to copy positions: {}", e)))?;

        // Allocate output buffer
        let mut d_moments = self
            .device
            .alloc_zeros::<f32>(num_steps)
            .map_err(|e| XError::GpuError(format!("Failed to allocate moments: {}", e)))?;

        // Load and launch kernel
        let kernel = self.load_moments_kernel()?;
        let threads_per_block = 256;
        let num_blocks = (num_steps + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(
                    config,
                    (
                        &d_positions,
                        &mut d_moments,
                        order,
                        num_particles as i32,
                        num_steps as i32,
                    ),
                )
                .map_err(|e| XError::GpuError(format!("Failed to launch moments kernel: {}", e)))?;
        }

        // Synchronize and copy results back
        self.device
            .synchronize()
            .map_err(|e| XError::GpuError(format!("Failed to synchronize: {}", e)))?;

        let moments = self
            .device
            .dtoh_sync_copy(&d_moments)
            .map_err(|e| XError::GpuError(format!("Failed to copy moments: {}", e)))?;

        Ok(moments)
    }

    /// Load OU process kernel
    pub fn load_ou_process_kernel(&mut self) -> XResult<Arc<CudaFunction>> {
        if let Some(ref kernel) = self.ou_process {
            return Ok(kernel.clone());
        }

        self.load_kernel_from_file_or_source(
            "kernels/cuda/ou.cu",
            OU_PROCESS_KERNEL,
            "ou_process",
            &["simulate_ou_process_f32"],
        )?;

        let kernel = self
            .device
            .get_func("ou_process", "simulate_ou_process_f32")
            .map_err(|e| XError::GpuError(format!("Failed to get kernel function: {}", e)))?;

        self.ou_process = Some(kernel.clone());
        Ok(kernel)
    }

    /// Simulate OU process on GPU
    pub fn simulate_ou_f32(
        &mut self,
        params: &KernelLaunchParams,
        start_position: f32,
        theta: f32,
        mu: f32,
        sigma: f32,
        time_step: f32,
        seed: u64,
    ) -> XResult<Vec<f32>> {
        // Initialize cuRAND states
        let d_states = self.init_curand_states(params.num_particles, seed)?;

        // Allocate output buffer
        let output_size = params.num_particles * (params.num_steps + 1);
        let mut d_positions = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| XError::GpuError(format!("Failed to allocate positions: {}", e)))?;

        // Load and launch kernel
        let kernel = self.load_ou_process_kernel()?;
        let config = params.get_launch_config();

        unsafe {
            kernel
                .launch(
                    config,
                    (
                        &d_states,
                        &mut d_positions,
                        start_position,
                        theta,
                        mu,
                        sigma,
                        time_step,
                        params.num_steps as i32,
                        params.num_particles as i32,
                    ),
                )
                .map_err(|e| XError::GpuError(format!("Failed to launch OU kernel: {}", e)))?;
        }

        // Synchronize and copy results back
        self.device
            .synchronize()
            .map_err(|e| XError::GpuError(format!("Failed to synchronize: {}", e)))?;

        let positions = self
            .device
            .dtoh_sync_copy(&d_positions)
            .map_err(|e| XError::GpuError(format!("Failed to copy results: {}", e)))?;

        Ok(positions)
    }

    /// Load GBM kernel
    pub fn load_gbm_kernel(&mut self) -> XResult<Arc<CudaFunction>> {
        if let Some(ref kernel) = self.gbm {
            return Ok(kernel.clone());
        }

        self.load_kernel_from_file_or_source(
            "kernels/cuda/geometric_bm.cu",
            GBM_KERNEL,
            "gbm",
            &["simulate_gbm_f32"],
        )?;

        let kernel = self
            .device
            .get_func("gbm", "simulate_gbm_f32")
            .map_err(|e| XError::GpuError(format!("Failed to get kernel function: {}", e)))?;

        self.gbm = Some(kernel.clone());
        Ok(kernel)
    }

    /// Simulate GBM on GPU
    pub fn simulate_gbm_f32(
        &mut self,
        params: &KernelLaunchParams,
        start_position: f32,
        mu: f32,
        sigma: f32,
        time_step: f32,
        seed: u64,
    ) -> XResult<Vec<f32>> {
        // Initialize cuRAND states
        let d_states = self.init_curand_states(params.num_particles, seed)?;

        // Allocate output buffer
        let output_size = params.num_particles * (params.num_steps + 1);
        let mut d_positions = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| XError::GpuError(format!("Failed to allocate positions: {}", e)))?;

        // Load and launch kernel
        let kernel = self.load_gbm_kernel()?;
        let config = params.get_launch_config();

        unsafe {
            kernel
                .launch(
                    config,
                    (
                        &d_states,
                        &mut d_positions,
                        start_position,
                        mu,
                        sigma,
                        time_step,
                        params.num_steps as i32,
                        params.num_particles as i32,
                    ),
                )
                .map_err(|e| XError::GpuError(format!("Failed to launch GBM kernel: {}", e)))?;
        }

        // Synchronize and copy results back
        self.device
            .synchronize()
            .map_err(|e| XError::GpuError(format!("Failed to synchronize: {}", e)))?;

        let positions = self
            .device
            .dtoh_sync_copy(&d_positions)
            .map_err(|e| XError::GpuError(format!("Failed to copy results: {}", e)))?;

        Ok(positions)
    }

    /// Get optimal launch configuration
    pub fn get_launch_config(&self, num_particles: usize) -> LaunchConfig {
        let threads_per_block = 256;
        let num_blocks = (num_particles + threads_per_block - 1) / threads_per_block;

        LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_source_not_empty() {
        assert!(!BROWNIAN_MOTION_KERNEL.is_empty());
        assert!(!MOMENTS_KERNEL.is_empty());
        assert!(!OU_PROCESS_KERNEL.is_empty());
        assert!(!GBM_KERNEL.is_empty());
    }
}
