//! 所有进程的 CUDA 实现

use super::{CudaBackend, KernelLaunchParams, KernelManager};
use crate::XResult;
use cudarc::driver::sys::CUdeviceptr;

impl KernelManager {
    /// 模拟分数布朗运动 (FBM)
    pub fn simulate_fbm_f32(
        &mut self,
        params: &KernelLaunchParams,
        start_position: f32,
        hurst: f32,
        diffusion_coefficient: f32,
        time_step: f32,
        seed: u64,
    ) -> XResult<Vec<f32>> {
        let output_size = params.num_particles * (params.num_steps + 1);
        let states =
            self.init_curand_states(params.num_particles, seed, params.threads_per_block)?;
        let positions = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| crate::XError::GpuError(format!("Failed to allocate: {}", e)))?;

        let ptx = cuda_kernel::FBM_PTX;
        self.device
            .load_ptx(ptx.into(), "fbm", &["simulate_fbm_f32"])
            .map_err(|e| crate::XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        let func = self
            .device
            .get_func("fbm", "simulate_fbm_f32")
            .ok_or_else(|| crate::XError::GpuError("Failed to get function".to_string()))?;

        let cfg = params.launch_config();
        unsafe {
            func.launch(
                cfg,
                (
                    &(*states.device_ptr() as CUdeviceptr),
                    &(*positions.device_ptr() as CUdeviceptr),
                    &start_position,
                    &hurst,
                    &diffusion_coefficient,
                    &time_step,
                    &(params.num_steps as i32),
                    &(params.num_particles as i32),
                ),
            )
            .map_err(|e| crate::XError::GpuError(format!("Launch failed: {}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| crate::XError::GpuError(format!("Sync failed: {}", e)))?;
        self.device
            .dtoh_sync_copy(&positions)
            .map_err(|e| crate::XError::GpuError(format!("Copy failed: {}", e)))
    }

    /// 模拟 Levy 过程
    pub fn simulate_levy_f32(
        &mut self,
        params: &KernelLaunchParams,
        start_position: f32,
        alpha: f32,
        beta: f32,
        scale: f32,
        time_step: f32,
        seed: u64,
    ) -> XResult<Vec<f32>> {
        let output_size = params.num_particles * (params.num_steps + 1);
        let states =
            self.init_curand_states(params.num_particles, seed, params.threads_per_block)?;
        let positions = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| crate::XError::GpuError(format!("Failed to allocate: {}", e)))?;

        let ptx = cuda_kernel::LEVY_PTX;
        self.device
            .load_ptx(ptx.into(), "levy", &["simulate_levy_f32"])
            .map_err(|e| crate::XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        let func = self
            .device
            .get_func("levy", "simulate_levy_f32")
            .ok_or_else(|| crate::XError::GpuError("Failed to get function".to_string()))?;

        let cfg = params.launch_config();
        unsafe {
            func.launch(
                cfg,
                (
                    &(*states.device_ptr() as CUdeviceptr),
                    &(*positions.device_ptr() as CUdeviceptr),
                    &start_position,
                    &alpha,
                    &beta,
                    &scale,
                    &time_step,
                    &(params.num_steps as i32),
                    &(params.num_particles as i32),
                ),
            )
            .map_err(|e| crate::XError::GpuError(format!("Launch failed: {}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| crate::XError::GpuError(format!("Sync failed: {}", e)))?;
        self.device
            .dtoh_sync_copy(&positions)
            .map_err(|e| crate::XError::GpuError(format!("Copy failed: {}", e)))
    }

    /// 模拟 Langevin 过程
    pub fn simulate_langevin_f32(
        &mut self,
        params: &KernelLaunchParams,
        start_position: f32,
        friction: f32,
        temperature: f32,
        mass: f32,
        time_step: f32,
        seed: u64,
    ) -> XResult<Vec<f32>> {
        let output_size = params.num_particles * (params.num_steps + 1);
        let states =
            self.init_curand_states(params.num_particles, seed, params.threads_per_block)?;
        let positions = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| crate::XError::GpuError(format!("Failed to allocate: {}", e)))?;

        let ptx = cuda_kernel::LANGEVIN_PTX;
        self.device
            .load_ptx(ptx.into(), "langevin", &["simulate_langevin_f32"])
            .map_err(|e| crate::XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        let func = self
            .device
            .get_func("langevin", "simulate_langevin_f32")
            .ok_or_else(|| crate::XError::GpuError("Failed to get function".to_string()))?;

        let cfg = params.launch_config();
        unsafe {
            func.launch(
                cfg,
                (
                    &(*states.device_ptr() as CUdeviceptr),
                    &(*positions.device_ptr() as CUdeviceptr),
                    &start_position,
                    &friction,
                    &temperature,
                    &mass,
                    &time_step,
                    &(params.num_steps as i32),
                    &(params.num_particles as i32),
                ),
            )
            .map_err(|e| crate::XError::GpuError(format!("Launch failed: {}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| crate::XError::GpuError(format!("Sync failed: {}", e)))?;
        self.device
            .dtoh_sync_copy(&positions)
            .map_err(|e| crate::XError::GpuError(format!("Copy failed: {}", e)))
    }

    /// 模拟 Cauchy 过程
    pub fn simulate_cauchy_f32(
        &mut self,
        params: &KernelLaunchParams,
        start_position: f32,
        scale: f32,
        time_step: f32,
        seed: u64,
    ) -> XResult<Vec<f32>> {
        let output_size = params.num_particles * (params.num_steps + 1);
        let states =
            self.init_curand_states(params.num_particles, seed, params.threads_per_block)?;
        let positions = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| crate::XError::GpuError(format!("Failed to allocate: {}", e)))?;

        let ptx = cuda_kernel::CAUCHY_PTX;
        self.device
            .load_ptx(ptx.into(), "cauchy", &["simulate_cauchy_f32"])
            .map_err(|e| crate::XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        let func = self
            .device
            .get_func("cauchy", "simulate_cauchy_f32")
            .ok_or_else(|| crate::XError::GpuError("Failed to get function".to_string()))?;

        let cfg = params.launch_config();
        unsafe {
            func.launch(
                cfg,
                (
                    &(*states.device_ptr() as CUdeviceptr),
                    &(*positions.device_ptr() as CUdeviceptr),
                    &start_position,
                    &scale,
                    &time_step,
                    &(params.num_steps as i32),
                    &(params.num_particles as i32),
                ),
            )
            .map_err(|e| crate::XError::GpuError(format!("Launch failed: {}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| crate::XError::GpuError(format!("Sync failed: {}", e)))?;
        self.device
            .dtoh_sync_copy(&positions)
            .map_err(|e| crate::XError::GpuError(format!("Copy failed: {}", e)))
    }

    /// 模拟 Gamma 过程
    pub fn simulate_gamma_process_f32(
        &mut self,
        params: &KernelLaunchParams,
        start_position: f32,
        shape: f32,
        rate: f32,
        time_step: f32,
        seed: u64,
    ) -> XResult<Vec<f32>> {
        let output_size = params.num_particles * (params.num_steps + 1);
        let states =
            self.init_curand_states(params.num_particles, seed, params.threads_per_block)?;
        let positions = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| crate::XError::GpuError(format!("Failed to allocate: {}", e)))?;

        let ptx = cuda_kernel::GAMMA_PTX;
        self.device
            .load_ptx(ptx.into(), "gamma", &["simulate_gamma_process_f32"])
            .map_err(|e| crate::XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        let func = self
            .device
            .get_func("gamma", "simulate_gamma_process_f32")
            .ok_or_else(|| crate::XError::GpuError("Failed to get function".to_string()))?;

        let cfg = params.launch_config();
        unsafe {
            func.launch(
                cfg,
                (
                    &(*states.device_ptr() as CUdeviceptr),
                    &(*positions.device_ptr() as CUdeviceptr),
                    &start_position,
                    &shape,
                    &rate,
                    &time_step,
                    &(params.num_steps as i32),
                    &(params.num_particles as i32),
                ),
            )
            .map_err(|e| crate::XError::GpuError(format!("Launch failed: {}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| crate::XError::GpuError(format!("Sync failed: {}", e)))?;
        self.device
            .dtoh_sync_copy(&positions)
            .map_err(|e| crate::XError::GpuError(format!("Copy failed: {}", e)))
    }

    /// 模拟布朗桥
    pub fn simulate_brownian_bridge_f32(
        &mut self,
        params: &KernelLaunchParams,
        start_position: f32,
        end_position: f32,
        diffusion_coefficient: f32,
        time_step: f32,
        seed: u64,
    ) -> XResult<Vec<f32>> {
        let output_size = params.num_particles * (params.num_steps + 1);
        let states =
            self.init_curand_states(params.num_particles, seed, params.threads_per_block)?;
        let positions = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| crate::XError::GpuError(format!("Failed to allocate: {}", e)))?;

        let ptx = cuda_kernel::BROWNIAN_BRIDGE_PTX;
        self.device
            .load_ptx(
                ptx.into(),
                "brownian_bridge",
                &["simulate_brownian_bridge_f32"],
            )
            .map_err(|e| crate::XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        let func = self
            .device
            .get_func("brownian_bridge", "simulate_brownian_bridge_f32")
            .ok_or_else(|| crate::XError::GpuError("Failed to get function".to_string()))?;

        let cfg = params.launch_config();
        unsafe {
            func.launch(
                cfg,
                (
                    &(*states.device_ptr() as CUdeviceptr),
                    &(*positions.device_ptr() as CUdeviceptr),
                    &start_position,
                    &end_position,
                    &diffusion_coefficient,
                    &time_step,
                    &(params.num_steps as i32),
                    &(params.num_particles as i32),
                ),
            )
            .map_err(|e| crate::XError::GpuError(format!("Launch failed: {}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| crate::XError::GpuError(format!("Sync failed: {}", e)))?;
        self.device
            .dtoh_sync_copy(&positions)
            .map_err(|e| crate::XError::GpuError(format!("Copy failed: {}", e)))
    }

    /// 模拟布朗漂移
    pub fn simulate_brownian_excursion_f32(
        &mut self,
        params: &KernelLaunchParams,
        diffusion_coefficient: f32,
        time_step: f32,
        seed: u64,
    ) -> XResult<Vec<f32>> {
        let output_size = params.num_particles * (params.num_steps + 1);
        let states =
            self.init_curand_states(params.num_particles, seed, params.threads_per_block)?;
        let positions = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| crate::XError::GpuError(format!("Failed to allocate: {}", e)))?;

        let ptx = cuda_kernel::BROWNIAN_EXCURSION_PTX;
        self.device
            .load_ptx(
                ptx.into(),
                "brownian_excursion",
                &["simulate_brownian_excursion_f32"],
            )
            .map_err(|e| crate::XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        let func = self
            .device
            .get_func("brownian_excursion", "simulate_brownian_excursion_f32")
            .ok_or_else(|| crate::XError::GpuError("Failed to get function".to_string()))?;

        let cfg = params.launch_config();
        unsafe {
            func.launch(
                cfg,
                (
                    &(*states.device_ptr() as CUdeviceptr),
                    &(*positions.device_ptr() as CUdeviceptr),
                    &diffusion_coefficient,
                    &time_step,
                    &(params.num_steps as i32),
                    &(params.num_particles as i32),
                ),
            )
            .map_err(|e| crate::XError::GpuError(format!("Launch failed: {}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| crate::XError::GpuError(format!("Sync failed: {}", e)))?;
        self.device
            .dtoh_sync_copy(&positions)
            .map_err(|e| crate::XError::GpuError(format!("Copy failed: {}", e)))
    }

    /// 模拟布朗曲折
    pub fn simulate_brownian_meander_f32(
        &mut self,
        params: &KernelLaunchParams,
        diffusion_coefficient: f32,
        time_step: f32,
        seed: u64,
    ) -> XResult<Vec<f32>> {
        let output_size = params.num_particles * (params.num_steps + 1);
        let states =
            self.init_curand_states(params.num_particles, seed, params.threads_per_block)?;
        let positions = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| crate::XError::GpuError(format!("Failed to allocate: {}", e)))?;

        let ptx = cuda_kernel::BROWNIAN_MEANDER_PTX;
        self.device
            .load_ptx(
                ptx.into(),
                "brownian_meander",
                &["simulate_brownian_meander_f32"],
            )
            .map_err(|e| crate::XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        let func = self
            .device
            .get_func("brownian_meander", "simulate_brownian_meander_f32")
            .ok_or_else(|| crate::XError::GpuError("Failed to get function".to_string()))?;

        let cfg = params.launch_config();
        unsafe {
            func.launch(
                cfg,
                (
                    &(*states.device_ptr() as CUdeviceptr),
                    &(*positions.device_ptr() as CUdeviceptr),
                    &diffusion_coefficient,
                    &time_step,
                    &(params.num_steps as i32),
                    &(params.num_particles as i32),
                ),
            )
            .map_err(|e| crate::XError::GpuError(format!("Launch failed: {}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| crate::XError::GpuError(format!("Sync failed: {}", e)))?;
        self.device
            .dtoh_sync_copy(&positions)
            .map_err(|e| crate::XError::GpuError(format!("Copy failed: {}", e)))
    }

    /// 模拟 BNG
    pub fn simulate_bng_f32(
        &mut self,
        params: &KernelLaunchParams,
        start_position: f32,
        diffusion_coefficient: f32,
        noise_intensity: f32,
        time_step: f32,
        seed: u64,
    ) -> XResult<Vec<f32>> {
        let output_size = params.num_particles * (params.num_steps + 1);
        let states =
            self.init_curand_states(params.num_particles, seed, params.threads_per_block)?;
        let positions = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| crate::XError::GpuError(format!("Failed to allocate: {}", e)))?;

        let ptx = cuda_kernel::BNG_PTX;
        self.device
            .load_ptx(ptx.into(), "bng", &["simulate_bng_f32"])
            .map_err(|e| crate::XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        let func = self
            .device
            .get_func("bng", "simulate_bng_f32")
            .ok_or_else(|| crate::XError::GpuError("Failed to get function".to_string()))?;

        let cfg = params.launch_config();
        unsafe {
            func.launch(
                cfg,
                (
                    &(*states.device_ptr() as CUdeviceptr),
                    &(*positions.device_ptr() as CUdeviceptr),
                    &start_position,
                    &diffusion_coefficient,
                    &noise_intensity,
                    &time_step,
                    &(params.num_steps as i32),
                    &(params.num_particles as i32),
                ),
            )
            .map_err(|e| crate::XError::GpuError(format!("Launch failed: {}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| crate::XError::GpuError(format!("Sync failed: {}", e)))?;
        self.device
            .dtoh_sync_copy(&positions)
            .map_err(|e| crate::XError::GpuError(format!("Copy failed: {}", e)))
    }

    /// 模拟 Levy 游走
    pub fn simulate_levy_walk_f32(
        &mut self,
        params: &KernelLaunchParams,
        start_position: f32,
        alpha: f32,
        velocity: f32,
        time_step: f32,
        seed: u64,
    ) -> XResult<Vec<f32>> {
        let output_size = params.num_particles * (params.num_steps + 1);
        let states =
            self.init_curand_states(params.num_particles, seed, params.threads_per_block)?;
        let positions = self
            .device
            .alloc_zeros::<f32>(output_size)
            .map_err(|e| crate::XError::GpuError(format!("Failed to allocate: {}", e)))?;

        let ptx = cuda_kernel::LEVY_WALK_PTX;
        self.device
            .load_ptx(ptx.into(), "levy_walk", &["simulate_levy_walk_f32"])
            .map_err(|e| crate::XError::GpuError(format!("Failed to load PTX: {}", e)))?;

        let func = self
            .device
            .get_func("levy_walk", "simulate_levy_walk_f32")
            .ok_or_else(|| crate::XError::GpuError("Failed to get function".to_string()))?;

        let cfg = params.launch_config();
        unsafe {
            func.launch(
                cfg,
                (
                    &(*states.device_ptr() as CUdeviceptr),
                    &(*positions.device_ptr() as CUdeviceptr),
                    &start_position,
                    &alpha,
                    &velocity,
                    &time_step,
                    &(params.num_steps as i32),
                    &(params.num_particles as i32),
                ),
            )
            .map_err(|e| crate::XError::GpuError(format!("Launch failed: {}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| crate::XError::GpuError(format!("Sync failed: {}", e)))?;
        self.device
            .dtoh_sync_copy(&positions)
            .map_err(|e| crate::XError::GpuError(format!("Copy failed: {}", e)))
    }
}
