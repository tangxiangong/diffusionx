//! CUDA kernel management and launch utilities

use crate::{XError, XResult};

#[cfg(feature = "cuda")]
use cudarc::driver::result::stream::CudaStream;
#[cfg(feature = "cuda")]
use cudarc::driver::{
    CudaDevice, CudaFunction, DevicePtr, DevicePtrMut, LaunchAsync, LaunchConfig,
};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Parameters for launching CUDA kernels
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct KernelLaunchParams {
    /// Number of particles to simulate
    pub num_particles: usize,

    /// Number of time steps
    pub num_steps: usize,

    /// Threads per block
    pub threads_per_block: usize,
}

#[cfg(feature = "cuda")]
impl KernelLaunchParams {
    /// Create new kernel launch parameters
    pub fn new(num_particles: usize, num_steps: usize) -> Self {
        Self {
            num_particles,
            num_steps,
            threads_per_block: 256,
        }
    }

    /// Set threads per block
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads_per_block = threads;
        self
    }

    /// Calculate number of blocks needed
    pub fn num_blocks(&self) -> usize {
        (self.num_particles + self.threads_per_block - 1) / self.threads_per_block
    }

    /// Create LaunchConfig for cudarc
    pub fn launch_config(&self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.num_blocks() as u32, 1, 1),
            block_dim: (self.threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// Statistics computed on GPU
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct GpuStats {
    pub mean: Vec<f32>,
    pub msd: Vec<f32>,
    pub variance: Vec<f32>,
}

/// Manages CUDA kernels for stochastic process simulation
#[cfg(feature = "cuda")]
pub struct KernelManager {
    device: Arc<CudaDevice>,
}

#[cfg(feature = "cuda")]
impl KernelManager {
    /// Create a new kernel manager
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    /// Initialize cuRAND states for random number generation
    fn init_curand_states(
        &mut self,
        num_particles: usize,
        seed: u64,
        threads_per_block: usize,
    ) -> XResult<DevicePtrMut<cudarc::driver::sys::CUdeviceptr>> {
        use cudarc::driver::sys::CUdeviceptr;
        use cudarc::nvrtc::compile_ptx;

        // Allocate memory for cuRAND states (sizeof(curandState) = 48 bytes)
        let state_size = 48;
        let states_buffer_size = num_particles * state_size;
        let states = self
            .device
            .alloc_zeros::<u8>(states_buffer_size)
            .map_err(|e| XError::GpuError(format!("Failed to allocate cuRAND states: {}", e)))?;

        // Load init_curand_states kernel from bm.cu
        let ptx = cuda_kernel::BM_PTX;
        self.device
            .load_ptx(ptx.into(), "bm", &["init_curand_states"])
            .map_err(|e| XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        let func = self
            .device
            .get_func("bm", "init_curand_states")
            .ok_or_else(|| {
                XError::GpuError("Failed to get init_curand_states function".to_string())
            })?;

        let num_blocks = (num_particles + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Cast states to CUdeviceptr for kernel launch
        let states_ptr = *states.device_ptr() as CUdeviceptr;

        unsafe {
            func.launch(cfg, (&states_ptr, &seed, &(num_particles as i32)))
                .map_err(|e| {
                    XError::GpuError(format!("Failed to launch init_curand_states: {}", e))
                })?;
        }

        self.device
            .synchronize()
            .map_err(|e| XError::GpuError(format!("Failed to synchronize after init: {}", e)))?;

        // Return the states buffer (reinterpreted as opaque pointer)
        Ok(states.transmute())
    }

    /// Simulate Brownian motion (f32)
    pub fn simulate_bm_f32(
        &mut self,
        params: &KernelLaunchParams,
        start_position: f32,
        diffusion_coefficient: f32,
        time_step: f32,
        seed: u64,
    ) -> XResult<Vec<f32>> {
        use cudarc::driver::sys::CUdeviceptr;

        let output_size = params.num_particles * (params.num_steps + 1);

        // Initialize cuRAND states
        let states =
            self.init_curand_states(params.num_particles, seed, params.threads_per_block)?;

        // Allocate output buffer
        let positions = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| XError::GpuError(format!("Failed to allocate positions: {}", e)))?;

        // Load kernel
        let ptx = cuda_kernel::BM_PTX;
        self.device
            .load_ptx(ptx.into(), "bm", &["simulate_bm_f32"])
            .map_err(|e| XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        let func = self
            .device
            .get_func("bm", "simulate_bm_f32")
            .ok_or_else(|| {
                XError::GpuError("Failed to get simulate_bm_f32 function".to_string())
            })?;

        let cfg = params.launch_config();
        let states_ptr = *states.device_ptr() as CUdeviceptr;
        let positions_ptr = *positions.device_ptr() as CUdeviceptr;
        let num_steps = params.num_steps as i32;
        let num_particles = params.num_particles as i32;

        unsafe {
            func.launch(
                cfg,
                (
                    &states_ptr,
                    &positions_ptr,
                    &start_position,
                    &diffusion_coefficient,
                    &time_step,
                    &num_steps,
                    &num_particles,
                ),
            )
            .map_err(|e| XError::GpuError(format!("Failed to launch simulate_bm_f32: {}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| XError::GpuError(format!("Failed to synchronize: {}", e)))?;

        // Copy results back to host
        let result = self
            .device
            .dtoh_sync_copy(&positions)
            .map_err(|e| XError::GpuError(format!("Failed to copy results: {}", e)))?;

        Ok(result)
    }

    /// Simulate Ornstein-Uhlenbeck process (f32)
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
        use cudarc::driver::sys::CUdeviceptr;

        let output_size = params.num_particles * (params.num_steps + 1);

        let states =
            self.init_curand_states(params.num_particles, seed, params.threads_per_block)?;

        let positions = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| XError::GpuError(format!("Failed to allocate positions: {}", e)))?;

        let ptx = cuda_kernel::OU_PTX;
        self.device
            .load_ptx(ptx.into(), "ou", &["simulate_ou_process_exact_f32"])
            .map_err(|e| XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        let func = self
            .device
            .get_func("ou", "simulate_ou_process_exact_f32")
            .ok_or_else(|| {
                XError::GpuError("Failed to get simulate_ou_process_exact_f32 function".to_string())
            })?;

        let cfg = params.launch_config();
        let states_ptr = *states.device_ptr() as CUdeviceptr;
        let positions_ptr = *positions.device_ptr() as CUdeviceptr;
        let num_steps = params.num_steps as i32;
        let num_particles = params.num_particles as i32;

        unsafe {
            func.launch(
                cfg,
                (
                    &states_ptr,
                    &positions_ptr,
                    &start_position,
                    &theta,
                    &mu,
                    &sigma,
                    &time_step,
                    &num_steps,
                    &num_particles,
                ),
            )
            .map_err(|e| {
                XError::GpuError(format!(
                    "Failed to launch simulate_ou_process_exact_f32: {}",
                    e
                ))
            })?;
        }

        self.device
            .synchronize()
            .map_err(|e| XError::GpuError(format!("Failed to synchronize: {}", e)))?;

        let result = self
            .device
            .dtoh_sync_copy(&positions)
            .map_err(|e| XError::GpuError(format!("Failed to copy results: {}", e)))?;

        Ok(result)
    }

    /// Simulate Geometric Brownian Motion (f32)
    pub fn simulate_gbm_f32(
        &mut self,
        params: &KernelLaunchParams,
        start_position: f32,
        mu: f32,
        sigma: f32,
        time_step: f32,
        seed: u64,
    ) -> XResult<Vec<f32>> {
        use cudarc::driver::sys::CUdeviceptr;

        let output_size = params.num_particles * (params.num_steps + 1);

        let states =
            self.init_curand_states(params.num_particles, seed, params.threads_per_block)?;

        let positions = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| XError::GpuError(format!("Failed to allocate positions: {}", e)))?;

        let ptx = cuda_kernel::GBM_PTX;
        self.device
            .load_ptx(ptx.into(), "gbm", &["simulate_geometric_bm_f32"])
            .map_err(|e| XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        let func = self
            .device
            .get_func("gbm", "simulate_geometric_bm_f32")
            .ok_or_else(|| {
                XError::GpuError("Failed to get simulate_geometric_bm_f32 function".to_string())
            })?;

        let cfg = params.launch_config();
        let states_ptr = *states.device_ptr() as CUdeviceptr;
        let positions_ptr = *positions.device_ptr() as CUdeviceptr;
        let num_steps = params.num_steps as i32;
        let num_particles = params.num_particles as i32;

        unsafe {
            func.launch(
                cfg,
                (
                    &states_ptr,
                    &positions_ptr,
                    &start_position,
                    &mu,
                    &sigma,
                    &time_step,
                    &num_steps,
                    &num_particles,
                ),
            )
            .map_err(|e| {
                XError::GpuError(format!("Failed to launch simulate_geometric_bm_f32: {}", e))
            })?;
        }

        self.device
            .synchronize()
            .map_err(|e| XError::GpuError(format!("Failed to synchronize: {}", e)))?;

        let result = self
            .device
            .dtoh_sync_copy(&positions)
            .map_err(|e| XError::GpuError(format!("Failed to copy results: {}", e)))?;

        Ok(result)
    }

    /// Generate Stable distributed random numbers on GPU
    pub fn generate_stable_f32(
        &mut self,
        n: usize,
        alpha: f32,
        beta: f32,
        seed: u64,
    ) -> XResult<Vec<f32>> {
        // 初始化 cuRAND 状态
        let threads_per_block = 256;
        let num_blocks = (n + threads_per_block - 1) / threads_per_block;

        let states = self.init_curand_states(n, seed, threads_per_block)?;
        let output = self
            .device
            .alloc_zeros::<f32>(n)
            .map_err(|e| crate::XError::GpuError(format!("Failed to allocate: {}", e)))?;

        // 加载 stable PTX
        let ptx = cuda_kernel::STABLE_PTX;
        self.device
            .load_ptx(ptx.into(), "stable", &["generate_stable_f32"])
            .map_err(|e| crate::XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        let func = self
            .device
            .get_func("stable", "generate_stable_f32")
            .ok_or_else(|| crate::XError::GpuError("Failed to get function".to_string()))?;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (
                    &(*states.device_ptr() as CUdeviceptr),
                    &(*output.device_ptr() as CUdeviceptr),
                    &alpha,
                    &beta,
                    &(n as i32),
                ),
            )
            .map_err(|e| crate::XError::GpuError(format!("Launch failed: {}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| crate::XError::GpuError(format!("Sync failed: {}", e)))?;
        self.device
            .dtoh_sync_copy(&output)
            .map_err(|e| crate::XError::GpuError(format!("Copy failed: {}", e)))
    }

    /// Compute Monte Carlo statistics on GPU
    pub fn compute_montecarlo_stats(
        &mut self,
        positions: &[f32],
        num_particles: usize,
        num_steps: usize,
    ) -> XResult<GpuStats> {
        use cudarc::driver::sys::CUdeviceptr;

        let output_size = num_steps + 1;

        // Upload positions to GPU
        let positions_gpu = self
            .device
            .htod_sync_copy(positions)
            .map_err(|e| XError::GpuError(format!("Failed to copy positions to device: {}", e)))?;

        // Allocate output buffers
        let mean_gpu = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| XError::GpuError(format!("Failed to allocate mean buffer: {}", e)))?;

        let msd_gpu = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| XError::GpuError(format!("Failed to allocate msd buffer: {}", e)))?;

        let variance_gpu = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| XError::GpuError(format!("Failed to allocate variance buffer: {}", e)))?;

        // Load kernel
        let ptx = cuda_kernel::STATS_PTX;
        self.device
            .load_ptx(ptx.into(), "stats", &["compute_stats_f32"])
            .map_err(|e| XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        let func = self
            .device
            .get_func("stats", "compute_stats_f32")
            .ok_or_else(|| {
                XError::GpuError("Failed to get compute_stats_f32 function".to_string())
            })?;

        let threads_per_block = 256;
        let num_blocks = (output_size + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let positions_ptr = *positions_gpu.device_ptr() as CUdeviceptr;
        let mean_ptr = *mean_gpu.device_ptr() as CUdeviceptr;
        let msd_ptr = *msd_gpu.device_ptr() as CUdeviceptr;
        let variance_ptr = *variance_gpu.device_ptr() as CUdeviceptr;
        let num_particles_i32 = num_particles as i32;
        let num_steps_i32 = num_steps as i32;

        unsafe {
            func.launch(
                cfg,
                (
                    &positions_ptr,
                    &mean_ptr,
                    &msd_ptr,
                    &variance_ptr,
                    &num_particles_i32,
                    &num_steps_i32,
                ),
            )
            .map_err(|e| XError::GpuError(format!("Failed to launch compute_stats_f32: {}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| XError::GpuError(format!("Failed to synchronize: {}", e)))?;

        // Copy results back
        let mean = self
            .device
            .dtoh_sync_copy(&mean_gpu)
            .map_err(|e| XError::GpuError(format!("Failed to copy mean: {}", e)))?;
        let msd = self
            .device
            .dtoh_sync_copy(&msd_gpu)
            .map_err(|e| XError::GpuError(format!("Failed to copy msd: {}", e)))?;
        let variance = self
            .device
            .dtoh_sync_copy(&variance_gpu)
            .map_err(|e| XError::GpuError(format!("Failed to copy variance: {}", e)))?;

        Ok(GpuStats {
            mean,
            msd,
            variance,
        })
    }
}
