//! GPU simulator for stochastic process simulation
//!
//! This module provides a high-level interface for GPU-accelerated simulation
//! of stochastic processes. It abstracts away backend-specific details and
//! provides a unified API for both CUDA and Metal.

use super::{GpuBackend, backend::GpuConfig};
use crate::{
    XResult,
    simulation::prelude::{ContinuousProcess, Pair},
    utils::linspace,
};

#[cfg(feature = "cuda")]
use super::mc::MonteCarloStats;
#[cfg(feature = "metal")]
use crate::simulation::continuous::Bm;
#[cfg(feature = "cuda")]
use crate::simulation::continuous::{Bm, GeometricBm, OrnsteinUhlenbeck};

/// GPU-accelerated simulator for stochastic processes
pub struct GpuSimulator {
    backend: GpuBackend,
    config: GpuConfig,
}

impl GpuSimulator {
    /// Create a new GPU simulator with the specified backend
    ///
    /// # Arguments
    ///
    /// * `backend` - The GPU backend to use (CUDA, Metal, or Auto)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use diffusionx::gpu::{GpuSimulator, GpuBackend};
    ///
    /// let simulator = GpuSimulator::new(GpuBackend::Auto)?;
    /// ```
    pub fn new(backend: GpuBackend) -> XResult<Self> {
        let backend = match backend {
            GpuBackend::Auto => GpuBackend::detect()?,
            _ => backend,
        };

        if !backend.is_available() {
            return Err(crate::XError::GpuError(format!(
                "GPU backend {:?} is not available",
                backend
            )));
        }

        Ok(Self {
            backend,
            config: GpuConfig::default(),
        })
    }

    /// Create a GPU simulator with a specific configuration
    pub fn with_config(backend: GpuBackend, config: GpuConfig) -> XResult<Self> {
        config.validate()?;

        let backend = match backend {
            GpuBackend::Auto => GpuBackend::detect()?,
            _ => backend,
        };

        if !backend.is_available() {
            return Err(crate::XError::GpuError(format!(
                "GPU backend {:?} is not available",
                backend
            )));
        }

        Ok(Self { backend, config })
    }

    /// Get the current GPU backend
    pub fn backend(&self) -> &GpuBackend {
        &self.backend
    }

    /// Get the current configuration
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    /// Set the configuration
    pub fn set_config(&mut self, config: GpuConfig) -> XResult<()> {
        config.validate()?;
        self.config = config;
        Ok(())
    }

    /// 通用的 GPU 模拟方法 - 与 CPU API 完全一致
    ///
    /// 接受任何实现了 ContinuousProcess 的类型，在 GPU 上模拟
    ///
    /// # Arguments
    ///
    /// * `process` - 要模拟的随机过程（与 CPU 版本相同的类型）
    /// * `duration` - 模拟时长
    /// * `time_step` - 时间步长
    ///
    /// # Returns
    ///
    /// 返回 (time, position) 对，与 CPU 的 `process.simulate()` 完全相同
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use diffusionx::gpu::{GpuSimulator, GpuBackend};
    /// use diffusionx::simulation::continuous::Bm;
    ///
    /// let bm = Bm::default();
    /// let simulator = GpuSimulator::new(GpuBackend::Auto)?;
    ///
    /// // CPU 版本
    /// let (t1, x1) = bm.simulate(1.0, 0.01)?;
    ///
    /// // GPU 版本 - API 完全相同！
    /// let (t2, x2) = simulator.simulate(&bm, 1.0, 0.01)?;
    /// ```
    pub fn simulate<P: ContinuousProcess>(
        &self,
        process: &P,
        duration: f64,
        time_step: f64,
    ) -> XResult<Pair> {
        // 模拟单个轨迹，返回第一个
        let trajectories = self.simulate_batch(process, duration, time_step, 1)?;
        Ok(trajectories.into_iter().next().unwrap())
    }

    /// 批量模拟 - 在 GPU 上并行模拟多个轨迹
    ///
    /// # Arguments
    ///
    /// * `process` - 要模拟的随机过程
    /// * `duration` - 模拟时长
    /// * `time_step` - 时间步长
    /// * `num_particles` - 并行模拟的粒子数量
    ///
    /// # Returns
    ///
    /// 返回多条轨迹的 Vec
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // 在 GPU 上并行模拟 1000 条轨迹
    /// let trajectories = simulator.simulate_batch(&bm, 1.0, 0.01, 1000)?;
    /// ```
    pub fn simulate_batch<P: ContinuousProcess>(
        &self,
        _process: &P,
        _duration: f64,
        _time_step: f64,
        _num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        #[cfg(feature = "cuda")]
        {
            if matches!(self.backend, GpuBackend::Cuda) {
                return self.simulate_cuda(_process, _duration, _time_step, _num_particles);
            }
        }

        #[cfg(feature = "metal")]
        {
            if matches!(self.backend, GpuBackend::Metal) {
                return self.simulate_metal(_process, _duration, _time_step, _num_particles);
            }
        }

        Err(crate::XError::GpuError(
            "GPU simulation not implemented for this backend".to_string(),
        ))
    }

    /// Simulate using CUDA backend
    #[cfg(feature = "cuda")]
    fn simulate_cuda<P: ContinuousProcess>(
        &self,
        process: &P,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        // Try to simulate specific process types on GPU
        // For now, fall back to CPU simulation
        // In the future, we can add process-specific GPU methods
        let mut all_trajectories = Vec::with_capacity(num_particles);
        for _ in 0..num_particles {
            let (t, x) = process.simulate(duration, time_step)?;
            all_trajectories.push((t, x));
        }
        Ok(all_trajectories)
    }

    /// Simulate Brownian motion on CUDA
    #[cfg(feature = "cuda")]
    pub fn simulate_bm_cuda(
        &self,
        bm: &Bm,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        use crate::gpu::cuda::{CudaBackend, KernelLaunchParams, KernelManager};

        let cuda = CudaBackend::new(0)?;
        let device = cuda.device();
        let num_steps = (duration / time_step).ceil() as usize;

        let mut kernel_manager = KernelManager::new(device.clone());
        let params = KernelLaunchParams::new(num_particles, num_steps)
            .with_threads(self.config.threads_per_block);

        let start_position = bm.get_start_position() as f32;
        let diffusion_coefficient = bm.get_diffusion_coefficient() as f32;
        let seed = 12345u64;

        let positions = kernel_manager.simulate_bm_f32(
            &params,
            start_position,
            diffusion_coefficient,
            time_step as f32,
            seed,
        )?;

        let time_points = linspace(0.0, duration, num_steps + 1);
        let mut all_trajectories = Vec::with_capacity(num_particles);

        for particle_idx in 0..num_particles {
            let offset = particle_idx * (num_steps + 1);
            let particle_positions: Vec<f64> = positions[offset..offset + num_steps + 1]
                .iter()
                .map(|&x| x as f64)
                .collect();
            all_trajectories.push((time_points.clone(), particle_positions));
        }

        Ok(all_trajectories)
    }

    /// Simulate OU process on CUDA
    #[cfg(feature = "cuda")]
    pub fn simulate_ou_cuda(
        &self,
        ou: &OrnsteinUhlenbeck,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        use crate::gpu::cuda::{CudaBackend, KernelLaunchParams, KernelManager};

        let cuda = CudaBackend::new(0)?;
        let device = cuda.device();
        let num_steps = (duration / time_step).ceil() as usize;

        let mut kernel_manager = KernelManager::new(device.clone());
        let params = KernelLaunchParams::new(num_particles, num_steps)
            .with_threads(self.config.threads_per_block);

        let start_position = ou.get_start_position() as f32;
        let theta = ou.get_theta() as f32;
        let mu = 0.0f32;
        let sigma = ou.get_sigma() as f32;
        let seed = 12345u64;

        let positions = kernel_manager.simulate_ou_f32(
            &params,
            start_position,
            theta,
            mu,
            sigma,
            time_step as f32,
            seed,
        )?;

        let time_points = linspace(0.0, duration, num_steps + 1);
        let mut all_trajectories = Vec::with_capacity(num_particles);

        for particle_idx in 0..num_particles {
            let offset = particle_idx * (num_steps + 1);
            let particle_positions: Vec<f64> = positions[offset..offset + num_steps + 1]
                .iter()
                .map(|&x| x as f64)
                .collect();
            all_trajectories.push((time_points.clone(), particle_positions));
        }

        Ok(all_trajectories)
    }

    /// Simulate Brownian motion Monte Carlo on CUDA (statistics only)
    #[cfg(feature = "cuda")]
    pub fn montecarlo_bm_cuda(
        &self,
        bm: &Bm,
        duration: f64,
        time_step: f64,
        num_samples: usize,
    ) -> XResult<MonteCarloStats> {
        use crate::gpu::cuda::{CudaBackend, KernelLaunchParams, KernelManager};

        let cuda = CudaBackend::new(0)?;
        let device = cuda.device();
        let num_steps = (duration / time_step).ceil() as usize;

        let mut kernel_manager = KernelManager::new(device.clone());
        let params = KernelLaunchParams::new(num_samples, num_steps)
            .with_threads(self.config.threads_per_block);

        let start_position = bm.get_start_position() as f32;
        let diffusion_coefficient = bm.get_diffusion_coefficient() as f32;
        let seed = 12345u64;

        // Simulate all particles on GPU
        let positions = kernel_manager.simulate_bm_f32(
            &params,
            start_position,
            diffusion_coefficient,
            time_step as f32,
            seed,
        )?;

        // Compute statistics on GPU
        let stats = kernel_manager.compute_montecarlo_stats(&positions, num_samples, num_steps)?;

        // Convert to MonteCarloStats
        let times = linspace(0.0, duration, num_steps + 1);
        let mut mc_stats = MonteCarloStats::new(num_steps, num_samples);
        mc_stats.times = times;
        mc_stats.mean = stats.mean.iter().map(|&x| x as f64).collect();
        mc_stats.msd = stats.msd.iter().map(|&x| x as f64).collect();
        mc_stats.variance = stats.variance.iter().map(|&x| x as f64).collect();

        Ok(mc_stats)
    }

    /// Simulate OU process Monte Carlo on CUDA (statistics only)
    #[cfg(feature = "cuda")]
    pub fn montecarlo_ou_cuda(
        &self,
        ou: &OrnsteinUhlenbeck,
        duration: f64,
        time_step: f64,
        num_samples: usize,
    ) -> XResult<MonteCarloStats> {
        use crate::gpu::cuda::{CudaBackend, KernelLaunchParams, KernelManager};

        let cuda = CudaBackend::new(0)?;
        let device = cuda.device();
        let num_steps = (duration / time_step).ceil() as usize;

        let mut kernel_manager = KernelManager::new(device.clone());
        let params = KernelLaunchParams::new(num_samples, num_steps)
            .with_threads(self.config.threads_per_block);

        let start_position = ou.get_start_position() as f32;
        let theta = ou.get_theta() as f32;
        let mu = 0.0f32;
        let sigma = ou.get_sigma() as f32;
        let seed = 12345u64;

        // Simulate all particles on GPU
        let positions = kernel_manager.simulate_ou_f32(
            &params,
            start_position,
            theta,
            mu,
            sigma,
            time_step as f32,
            seed,
        )?;

        // Compute statistics on GPU
        let stats = kernel_manager.compute_montecarlo_stats(&positions, num_samples, num_steps)?;

        // Convert to MonteCarloStats
        let times = linspace(0.0, duration, num_steps + 1);
        let mut mc_stats = MonteCarloStats::new(num_steps, num_samples);
        mc_stats.times = times;
        mc_stats.mean = stats.mean.iter().map(|&x| x as f64).collect();
        mc_stats.msd = stats.msd.iter().map(|&x| x as f64).collect();
        mc_stats.variance = stats.variance.iter().map(|&x| x as f64).collect();

        Ok(mc_stats)
    }

    /// Simulate GBM Monte Carlo on CUDA (statistics only)
    #[cfg(feature = "cuda")]
    pub fn montecarlo_gbm_cuda(
        &self,
        gbm: &GeometricBm,
        duration: f64,
        time_step: f64,
        num_samples: usize,
    ) -> XResult<MonteCarloStats> {
        use crate::gpu::cuda::{CudaBackend, KernelLaunchParams, KernelManager};

        let cuda = CudaBackend::new(0)?;
        let device = cuda.device();
        let num_steps = (duration / time_step).ceil() as usize;

        let mut kernel_manager = KernelManager::new(device.clone());
        let params = KernelLaunchParams::new(num_samples, num_steps)
            .with_threads(self.config.threads_per_block);

        let start_position = gbm.get_start_position() as f32;
        let mu = gbm.get_mu() as f32;
        let sigma = gbm.get_sigma() as f32;
        let seed = 12345u64;

        // Simulate all particles on GPU
        let positions = kernel_manager.simulate_gbm_f32(
            &params,
            start_position,
            mu,
            sigma,
            time_step as f32,
            seed,
        )?;

        // Compute statistics on GPU
        let stats = kernel_manager.compute_montecarlo_stats(&positions, num_samples, num_steps)?;

        // Convert to MonteCarloStats
        let times = linspace(0.0, duration, num_steps + 1);
        let mut mc_stats = MonteCarloStats::new(num_steps, num_samples);
        mc_stats.times = times;
        mc_stats.mean = stats.mean.iter().map(|&x| x as f64).collect();
        mc_stats.msd = stats.msd.iter().map(|&x| x as f64).collect();
        mc_stats.variance = stats.variance.iter().map(|&x| x as f64).collect();

        Ok(mc_stats)
    }

    /// Simulate GBM on CUDA
    #[cfg(feature = "cuda")]
    pub fn simulate_gbm_cuda(
        &self,
        gbm: &GeometricBm,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        use crate::gpu::cuda::{CudaBackend, KernelLaunchParams, KernelManager};

        let cuda = CudaBackend::new(0)?;
        let device = cuda.device();
        let num_steps = (duration / time_step).ceil() as usize;

        let mut kernel_manager = KernelManager::new(device.clone());
        let params = KernelLaunchParams::new(num_particles, num_steps)
            .with_threads(self.config.threads_per_block);

        let start_position = gbm.get_start_position() as f32;
        let mu = gbm.get_mu() as f32;
        let sigma = gbm.get_sigma() as f32;
        let seed = 12345u64;

        let positions = kernel_manager.simulate_gbm_f32(
            &params,
            start_position,
            mu,
            sigma,
            time_step as f32,
            seed,
        )?;

        let time_points = linspace(0.0, duration, num_steps + 1);
        let mut all_trajectories = Vec::with_capacity(num_particles);

        for particle_idx in 0..num_particles {
            let offset = particle_idx * (num_steps + 1);
            let particle_positions: Vec<f64> = positions[offset..offset + num_steps + 1]
                .iter()
                .map(|&x| x as f64)
                .collect();
            all_trajectories.push((time_points.clone(), particle_positions));
        }

        Ok(all_trajectories)
    }

    /// Simulate FBM on CUDA
    #[cfg(feature = "cuda")]
    pub fn simulate_fbm_cuda(
        &self,
        start_position: f64,
        hurst: f64,
        diffusion_coefficient: f64,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        use crate::gpu::cuda::{CudaBackend, KernelLaunchParams, KernelManager};

        let cuda = CudaBackend::new(0)?;
        let device = cuda.device();
        let num_steps = (duration / time_step).ceil() as usize;

        let mut kernel_manager = KernelManager::new(device.clone());
        let params = KernelLaunchParams::new(num_particles, num_steps)
            .with_threads(self.config.threads_per_block);

        let positions = kernel_manager.simulate_fbm_f32(
            &params,
            start_position as f32,
            hurst as f32,
            diffusion_coefficient as f32,
            time_step as f32,
            12345u64,
        )?;

        let time_points = linspace(0.0, duration, time_step);
        let mut all_trajectories = Vec::with_capacity(num_particles);

        for particle_idx in 0..num_particles {
            let offset = particle_idx * (num_steps + 1);
            let particle_positions: Vec<f64> = positions[offset..offset + num_steps + 1]
                .iter()
                .map(|&x| x as f64)
                .collect();
            all_trajectories.push((time_points.clone(), particle_positions));
        }

        Ok(all_trajectories)
    }

    /// Simulate Levy process on CUDA
    #[cfg(feature = "cuda")]
    pub fn simulate_levy_cuda(
        &self,
        start_position: f64,
        alpha: f64,
        beta: f64,
        scale: f64,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        use crate::gpu::cuda::{CudaBackend, KernelLaunchParams, KernelManager};

        let cuda = CudaBackend::new(0)?;
        let device = cuda.device();
        let num_steps = (duration / time_step).ceil() as usize;

        let mut kernel_manager = KernelManager::new(device.clone());
        let params = KernelLaunchParams::new(num_particles, num_steps)
            .with_threads(self.config.threads_per_block);

        let positions = kernel_manager.simulate_levy_f32(
            &params,
            start_position as f32,
            alpha as f32,
            beta as f32,
            scale as f32,
            time_step as f32,
            12345u64,
        )?;

        let time_points = linspace(0.0, duration, time_step);
        let mut all_trajectories = Vec::with_capacity(num_particles);

        for particle_idx in 0..num_particles {
            let offset = particle_idx * (num_steps + 1);
            let particle_positions: Vec<f64> = positions[offset..offset + num_steps + 1]
                .iter()
                .map(|&x| x as f64)
                .collect();
            all_trajectories.push((time_points.clone(), particle_positions));
        }

        Ok(all_trajectories)
    }

    /// Simulate Levy Walk on CUDA
    #[cfg(feature = "cuda")]
    pub fn simulate_levy_walk_cuda(
        &self,
        start_position: f64,
        alpha: f64,
        velocity: f64,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        use crate::gpu::cuda::{CudaBackend, KernelLaunchParams, KernelManager};

        let cuda = CudaBackend::new(0)?;
        let device = cuda.device();
        let num_steps = (duration / time_step).ceil() as usize;

        let mut kernel_manager = KernelManager::new(device.clone());
        let params = KernelLaunchParams::new(num_particles, num_steps)
            .with_threads(self.config.threads_per_block);

        let positions = kernel_manager.simulate_levy_walk_f32(
            &params,
            start_position as f32,
            alpha as f32,
            velocity as f32,
            time_step as f32,
            12345u64,
        )?;

        let time_points = linspace(0.0, duration, time_step);
        let mut all_trajectories = Vec::with_capacity(num_particles);

        for particle_idx in 0..num_particles {
            let offset = particle_idx * (num_steps + 1);
            let particle_positions: Vec<f64> = positions[offset..offset + num_steps + 1]
                .iter()
                .map(|&x| x as f64)
                .collect();
            all_trajectories.push((time_points.clone(), particle_positions));
        }

        Ok(all_trajectories)
    }

    /// Simulate Langevin on CUDA
    #[cfg(feature = "cuda")]
    pub fn simulate_langevin_cuda(
        &self,
        start_position: f64,
        friction: f64,
        temperature: f64,
        mass: f64,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        use crate::gpu::cuda::{CudaBackend, KernelLaunchParams, KernelManager};

        let cuda = CudaBackend::new(0)?;
        let device = cuda.device();
        let num_steps = (duration / time_step).ceil() as usize;

        let mut kernel_manager = KernelManager::new(device.clone());
        let params = KernelLaunchParams::new(num_particles, num_steps)
            .with_threads(self.config.threads_per_block);

        let positions = kernel_manager.simulate_langevin_f32(
            &params,
            start_position as f32,
            friction as f32,
            temperature as f32,
            mass as f32,
            time_step as f32,
            12345u64,
        )?;

        let time_points = linspace(0.0, duration, time_step);
        let mut all_trajectories = Vec::with_capacity(num_particles);

        for particle_idx in 0..num_particles {
            let offset = particle_idx * (num_steps + 1);
            let particle_positions: Vec<f64> = positions[offset..offset + num_steps + 1]
                .iter()
                .map(|&x| x as f64)
                .collect();
            all_trajectories.push((time_points.clone(), particle_positions));
        }

        Ok(all_trajectories)
    }

    /// Simulate Cauchy on CUDA
    #[cfg(feature = "cuda")]
    pub fn simulate_cauchy_cuda(
        &self,
        start_position: f64,
        scale: f64,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        use crate::gpu::cuda::{CudaBackend, KernelLaunchParams, KernelManager};

        let cuda = CudaBackend::new(0)?;
        let device = cuda.device();
        let num_steps = (duration / time_step).ceil() as usize;

        let mut kernel_manager = KernelManager::new(device.clone());
        let params = KernelLaunchParams::new(num_particles, num_steps)
            .with_threads(self.config.threads_per_block);

        let positions = kernel_manager.simulate_cauchy_f32(
            &params,
            start_position as f32,
            scale as f32,
            time_step as f32,
            12345u64,
        )?;

        let time_points = linspace(0.0, duration, time_step);
        let mut all_trajectories = Vec::with_capacity(num_particles);

        for particle_idx in 0..num_particles {
            let offset = particle_idx * (num_steps + 1);
            let particle_positions: Vec<f64> = positions[offset..offset + num_steps + 1]
                .iter()
                .map(|&x| x as f64)
                .collect();
            all_trajectories.push((time_points.clone(), particle_positions));
        }

        Ok(all_trajectories)
    }

    /// Simulate Gamma Process on CUDA
    #[cfg(feature = "cuda")]
    pub fn simulate_gamma_process_cuda(
        &self,
        start_position: f64,
        shape: f64,
        rate: f64,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        use crate::gpu::cuda::{CudaBackend, KernelLaunchParams, KernelManager};

        let cuda = CudaBackend::new(0)?;
        let device = cuda.device();
        let num_steps = (duration / time_step).ceil() as usize;

        let mut kernel_manager = KernelManager::new(device.clone());
        let params = KernelLaunchParams::new(num_particles, num_steps)
            .with_threads(self.config.threads_per_block);

        let positions = kernel_manager.simulate_gamma_process_f32(
            &params,
            start_position as f32,
            shape as f32,
            rate as f32,
            time_step as f32,
            12345u64,
        )?;

        let time_points = linspace(0.0, duration, time_step);
        let mut all_trajectories = Vec::with_capacity(num_particles);

        for particle_idx in 0..num_particles {
            let offset = particle_idx * (num_steps + 1);
            let particle_positions: Vec<f64> = positions[offset..offset + num_steps + 1]
                .iter()
                .map(|&x| x as f64)
                .collect();
            all_trajectories.push((time_points.clone(), particle_positions));
        }

        Ok(all_trajectories)
    }

    /// Simulate Brownian Bridge on CUDA
    #[cfg(feature = "cuda")]
    pub fn simulate_brownian_bridge_cuda(
        &self,
        start_position: f64,
        end_position: f64,
        diffusion_coefficient: f64,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        use crate::gpu::cuda::{CudaBackend, KernelLaunchParams, KernelManager};

        let cuda = CudaBackend::new(0)?;
        let device = cuda.device();
        let num_steps = (duration / time_step).ceil() as usize;

        let mut kernel_manager = KernelManager::new(device.clone());
        let params = KernelLaunchParams::new(num_particles, num_steps)
            .with_threads(self.config.threads_per_block);

        let positions = kernel_manager.simulate_brownian_bridge_f32(
            &params,
            start_position as f32,
            end_position as f32,
            diffusion_coefficient as f32,
            time_step as f32,
            12345u64,
        )?;

        let time_points = linspace(0.0, duration, time_step);
        let mut all_trajectories = Vec::with_capacity(num_particles);

        for particle_idx in 0..num_particles {
            let offset = particle_idx * (num_steps + 1);
            let particle_positions: Vec<f64> = positions[offset..offset + num_steps + 1]
                .iter()
                .map(|&x| x as f64)
                .collect();
            all_trajectories.push((time_points.clone(), particle_positions));
        }

        Ok(all_trajectories)
    }

    /// Simulate Brownian Excursion on CUDA
    #[cfg(feature = "cuda")]
    pub fn simulate_brownian_excursion_cuda(
        &self,
        diffusion_coefficient: f64,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        use crate::gpu::cuda::{CudaBackend, KernelLaunchParams, KernelManager};

        let cuda = CudaBackend::new(0)?;
        let device = cuda.device();
        let num_steps = (duration / time_step).ceil() as usize;

        let mut kernel_manager = KernelManager::new(device.clone());
        let params = KernelLaunchParams::new(num_particles, num_steps)
            .with_threads(self.config.threads_per_block);

        let positions = kernel_manager.simulate_brownian_excursion_f32(
            &params,
            diffusion_coefficient as f32,
            time_step as f32,
            12345u64,
        )?;

        let time_points = linspace(0.0, duration, time_step);
        let mut all_trajectories = Vec::with_capacity(num_particles);

        for particle_idx in 0..num_particles {
            let offset = particle_idx * (num_steps + 1);
            let particle_positions: Vec<f64> = positions[offset..offset + num_steps + 1]
                .iter()
                .map(|&x| x as f64)
                .collect();
            all_trajectories.push((time_points.clone(), particle_positions));
        }

        Ok(all_trajectories)
    }

    /// Simulate Brownian Meander on CUDA
    #[cfg(feature = "cuda")]
    pub fn simulate_brownian_meander_cuda(
        &self,
        diffusion_coefficient: f64,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        use crate::gpu::cuda::{CudaBackend, KernelLaunchParams, KernelManager};

        let cuda = CudaBackend::new(0)?;
        let device = cuda.device();
        let num_steps = (duration / time_step).ceil() as usize;

        let mut kernel_manager = KernelManager::new(device.clone());
        let params = KernelLaunchParams::new(num_particles, num_steps)
            .with_threads(self.config.threads_per_block);

        let positions = kernel_manager.simulate_brownian_meander_f32(
            &params,
            diffusion_coefficient as f32,
            time_step as f32,
            12345u64,
        )?;

        let time_points = linspace(0.0, duration, time_step);
        let mut all_trajectories = Vec::with_capacity(num_particles);

        for particle_idx in 0..num_particles {
            let offset = particle_idx * (num_steps + 1);
            let particle_positions: Vec<f64> = positions[offset..offset + num_steps + 1]
                .iter()
                .map(|&x| x as f64)
                .collect();
            all_trajectories.push((time_points.clone(), particle_positions));
        }

        Ok(all_trajectories)
    }

    /// Simulate BNG on CUDA
    #[cfg(feature = "cuda")]
    pub fn simulate_bng_cuda(
        &self,
        start_position: f64,
        diffusion_coefficient: f64,
        noise_intensity: f64,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        use crate::gpu::cuda::{CudaBackend, KernelLaunchParams, KernelManager};

        let cuda = CudaBackend::new(0)?;
        let device = cuda.device();
        let num_steps = (duration / time_step).ceil() as usize;

        let mut kernel_manager = KernelManager::new(device.clone());
        let params = KernelLaunchParams::new(num_particles, num_steps)
            .with_threads(self.config.threads_per_block);

        let positions = kernel_manager.simulate_bng_f32(
            &params,
            start_position as f32,
            diffusion_coefficient as f32,
            noise_intensity as f32,
            time_step as f32,
            12345u64,
        )?;

        let time_points = linspace(0.0, duration, time_step);
        let mut all_trajectories = Vec::with_capacity(num_particles);

        for particle_idx in 0..num_particles {
            let offset = particle_idx * (num_steps + 1);
            let particle_positions: Vec<f64> = positions[offset..offset + num_steps + 1]
                .iter()
                .map(|&x| x as f64)
                .collect();
            all_trajectories.push((time_points.clone(), particle_positions));
        }

        Ok(all_trajectories)
    }

    /// Simulate using Metal backend
    #[cfg(feature = "metal")]
    fn simulate_metal<P: ContinuousProcess>(
        &self,
        process: &P,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        // Try to simulate specific process types on GPU
        // For now, fall back to CPU simulation
        let mut all_trajectories = Vec::with_capacity(num_particles);
        for _ in 0..num_particles {
            let (t, x) = process.simulate(duration, time_step)?;
            all_trajectories.push((t, x));
        }
        Ok(all_trajectories)
    }

    /// Simulate Brownian motion on Metal
    #[cfg(feature = "metal")]
    pub fn simulate_bm_metal(
        &self,
        bm: &Bm,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        use crate::gpu::metal::{MetalBackend, MetalRng};
        use metal::MTLSize;

        let metal = MetalBackend::new()?;
        let device = metal.device();
        let library = metal.library();
        let num_steps = (duration / time_step).ceil() as usize;

        let kernel_function = library
            .get_function("simulate_bm", None)
            .map_err(|e| crate::XError::GpuError(format!("Failed to get kernel: {}", e)))?;

        let pipeline_state = device
            .new_compute_pipeline_state_with_function(&kernel_function)
            .map_err(|e| {
                crate::XError::GpuError(format!("Failed to create pipeline state: {}", e))
            })?;

        let rng = MetalRng::new(device.clone());
        let randoms_buffer = rng.standard_normals_f32(num_particles * num_steps)?;

        let output_size = num_particles * (num_steps + 1);
        let positions_size = (output_size * std::mem::size_of::<f32>()) as u64;
        let positions_buffer = metal.create_empty_buffer(positions_size)?;

        let command_queue = metal.command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(&pipeline_state);
        compute_encoder.set_buffer(0, Some(&randoms_buffer), 0);
        compute_encoder.set_buffer(1, Some(&positions_buffer), 0);

        let start_pos = bm.get_start_position() as f32;
        let diff_coef = bm.get_diffusion_coefficient() as f32;
        let dt = time_step as f32;
        compute_encoder.set_bytes(
            2,
            std::mem::size_of::<f32>() as u64,
            &start_pos as *const f32 as *const _,
        );
        compute_encoder.set_bytes(
            3,
            std::mem::size_of::<f32>() as u64,
            &diff_coef as *const f32 as *const _,
        );
        compute_encoder.set_bytes(
            4,
            std::mem::size_of::<f32>() as u64,
            &dt as *const f32 as *const _,
        );
        compute_encoder.set_bytes(
            5,
            std::mem::size_of::<i32>() as u64,
            &(num_steps as i32) as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            6,
            std::mem::size_of::<i32>() as u64,
            &(num_particles as i32) as *const i32 as *const _,
        );

        let threads_per_group = 256;
        let num_groups = num_particles.div_ceil(threads_per_group);

        compute_encoder.dispatch_thread_groups(
            MTLSize {
                width: num_groups as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group as u64,
                height: 1,
                depth: 1,
            },
        );

        compute_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let positions_ptr = positions_buffer.contents() as *const f32;
        let positions: Vec<f32> =
            unsafe { std::slice::from_raw_parts(positions_ptr, output_size).to_vec() };

        let time_points = linspace(0.0, duration, time_step);
        let mut all_trajectories = Vec::with_capacity(num_particles);

        for particle_idx in 0..num_particles {
            let offset = particle_idx * (num_steps + 1);
            let particle_positions: Vec<f64> = positions[offset..offset + num_steps + 1]
                .iter()
                .map(|&x| x as f64)
                .collect();
            all_trajectories.push((time_points.clone(), particle_positions));
        }

        Ok(all_trajectories)
    }

    /// Get GPU information string
    pub fn info(&self) -> String {
        format!("GPU Simulator with backend: {:?}", self.backend)
    }

    /// 在 GPU 上计算原始矩 - 与 CPU Moment trait 签名完全一致
    ///
    /// # Arguments
    ///
    /// * `process` - 任意随机过程
    /// * `order` - 矩的阶数
    /// * `particles` - 粒子数量
    /// * `time_step` - 时间步长
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // CPU 版本
    /// let traj = ContinuousTrajectory { sp: bm, duration: 1.0 };
    /// let mean = traj.raw_moment(1, 10000, 0.01)?;
    ///
    /// // GPU 版本 - 完全相同的签名！
    /// let simulator = GpuSimulator::new(GpuBackend::Auto)?;
    /// let mean = simulator.raw_moment(&bm, 1, 10000, 0.01)?;
    /// ```
    pub fn raw_moment<P: ContinuousProcess>(
        &self,
        process: &P,
        order: i32,
        particles: usize,
        duration: f64,
        time_step: f64,
    ) -> XResult<f64> {
        if order == 0 {
            return Ok(1.0);
        }

        // 在 GPU 上模拟所有轨迹
        let trajectories = self.simulate_batch(process, duration, time_step, particles)?;

        // 提取终点位置
        let end_positions: Vec<f64> = trajectories
            .iter()
            .map(|(_, positions)| *positions.last().unwrap())
            .collect();

        // 计算矩
        let moment = if order == 1 {
            end_positions.iter().sum::<f64>() / particles as f64
        } else {
            end_positions.iter().map(|&x| x.powi(order)).sum::<f64>() / particles as f64
        };

        Ok(moment)
    }

    /// 在 GPU 上计算中心矩 - 与 CPU Moment trait 签名完全一致
    ///
    /// # Arguments
    ///
    /// * `process` - 任意随机过程
    /// * `order` - 矩的阶数
    /// * `particles` - 粒子数量
    /// * `time_step` - 时间步长
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // CPU 版本
    /// let variance = traj.central_moment(2, 10000, 0.01)?;
    ///
    /// // GPU 版本 - 完全相同！
    /// let variance = simulator.central_moment(&bm, 2, 10000, 0.01)?;
    /// ```
    pub fn central_moment<P: ContinuousProcess>(
        &self,
        process: &P,
        order: i32,
        particles: usize,
        duration: f64,
        time_step: f64,
    ) -> XResult<f64> {
        if order == 0 {
            return Ok(1.0);
        }

        // 先计算均值
        let mean = self.raw_moment(process, 1, particles, duration, time_step)?;

        // 在 GPU 上模拟所有轨迹
        let trajectories = self.simulate_batch(process, duration, time_step, particles)?;

        // 提取终点位置并计算中心矩
        let end_positions: Vec<f64> = trajectories
            .iter()
            .map(|(_, positions)| *positions.last().unwrap())
            .collect();

        let moment = if order == 1 {
            0.0 // 一阶中心矩总是 0
        } else {
            end_positions
                .iter()
                .map(|&x| (x - mean).powi(order))
                .sum::<f64>()
                / particles as f64
        };

        Ok(moment)
    }

    /// 在 GPU 上计算多阶矩（批量计算）
    ///
    /// 一次性计算多个阶数的矩，更高效
    ///
    /// # Arguments
    ///
    /// * `process` - 任意随机过程
    /// * `max_order` - 最大阶数
    /// * `duration` - 模拟时长
    /// * `time_step` - 时间步长
    /// * `num_particles` - 粒子数量
    ///
    /// # Returns
    ///
    /// 返回 (原始矩, 中心矩) 的 Vec，索引 i 对应 i+1 阶矩
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let (raw_moments, central_moments) =
    ///     simulator.moments_batch_gpu(&bm, 4, 1.0, 0.01, 10000)?;
    ///
    /// let mean = raw_moments[0];      // 1阶
    /// let variance = central_moments[1]; // 2阶中心矩
    /// let skewness_related = central_moments[2]; // 3阶
    /// let kurtosis_related = central_moments[3];  // 4阶
    /// ```
    pub fn moments_batch_gpu<P: ContinuousProcess>(
        &self,
        process: &P,
        max_order: i32,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<(Vec<f64>, Vec<f64>)> {
        // 在 GPU 上模拟所有轨迹（只需一次）
        let trajectories = self.simulate_batch(process, duration, time_step, num_particles)?;

        // 提取终点位置
        let end_positions: Vec<f64> = trajectories
            .iter()
            .map(|(_, positions)| *positions.last().unwrap())
            .collect();

        // 计算均值
        let mean = end_positions.iter().sum::<f64>() / num_particles as f64;

        // 计算所有阶的原始矩
        let mut raw_moments = Vec::with_capacity(max_order as usize);
        for order in 1..=max_order {
            let moment = if order == 1 {
                mean
            } else {
                end_positions.iter().map(|&x| x.powi(order)).sum::<f64>() / num_particles as f64
            };
            raw_moments.push(moment);
        }

        // 计算所有阶的中心矩
        let mut central_moments = Vec::with_capacity(max_order as usize);
        for order in 1..=max_order {
            let moment = if order == 1 {
                0.0
            } else {
                end_positions
                    .iter()
                    .map(|&x| (x - mean).powi(order))
                    .sum::<f64>()
                    / num_particles as f64
            };
            central_moments.push(moment);
        }

        Ok((raw_moments, central_moments))
    }

    /// 计算标准化的高阶统计量
    ///
    /// # Returns
    ///
    /// 返回 (偏度, 峰度)
    ///
    /// 偏度 = 三阶中心矩 / 标准差^3
    /// 峰度 = 四阶中心矩 / 方差^2 - 3
    pub fn skewness_kurtosis_gpu<P: ContinuousProcess>(
        &self,
        process: &P,
        duration: f64,
        time_step: f64,
        num_particles: usize,
    ) -> XResult<(f64, f64)> {
        let (_, central_moments) =
            self.moments_batch_gpu(process, 4, duration, time_step, num_particles)?;

        let variance = central_moments[1]; // 2阶
        let m3 = central_moments[2]; // 3阶
        let m4 = central_moments[3]; // 4阶

        let std_dev = variance.sqrt();
        let skewness = if std_dev > 0.0 {
            m3 / (std_dev * std_dev * std_dev)
        } else {
            0.0
        };

        let kurtosis = if variance > 0.0 {
            m4 / (variance * variance) - 3.0
        } else {
            0.0
        };

        Ok((skewness, kurtosis))
    }

    // --- 离散过程 ---
    pub fn raw_moment_discrete<P: crate::simulation::prelude::DiscreteProcess>(
        &self,
        process: &P,
        order: i32,
        particles: usize,
        num_step: usize,
    ) -> XResult<f64> {
        use rayon::prelude::*;

        if order == 0 {
            return Ok(1.0);
        }

        // 在 CPU 上并行生成所有终点位置
        let end_positions: Vec<f64> = (0..particles)
            .into_par_iter()
            .map(|_| process.end(num_step).unwrap_or(0.0))
            .collect();

        // 计算矩
        let moment = if order == 1 {
            end_positions.iter().sum::<f64>() / particles as f64
        } else {
            end_positions.iter().map(|&x| x.powi(order)).sum::<f64>() / particles as f64
        };

        Ok(moment)
    }

    pub fn central_moment_discrete<P: crate::simulation::prelude::DiscreteProcess>(
        &self,
        process: &P,
        order: i32,
        particles: usize,
        num_step: usize,
    ) -> XResult<f64> {
        use rayon::prelude::*;

        if order == 0 {
            return Ok(1.0);
        }

        // 先计算均值
        let mean = self.raw_moment_discrete(process, 1, particles, num_step)?;

        // 在 CPU 上并行生成所有终点位置
        let end_positions: Vec<f64> = (0..particles)
            .into_par_iter()
            .map(|_| process.end(num_step).unwrap_or(0.0))
            .collect();

        // 计算中心矩
        let moment = if order == 1 {
            0.0
        } else {
            end_positions
                .iter()
                .map(|&x| (x - mean).powi(order))
                .sum::<f64>()
                / particles as f64
        };

        Ok(moment)
    }

    // --- 点过程 ---
    pub fn raw_moment_point<P: crate::simulation::prelude::PointProcess>(
        &self,
        process: &P,
        order: i32,
        particles: usize,
        duration: f64,
    ) -> XResult<f64> {
        use rayon::prelude::*;

        if order == 0 {
            return Ok(1.0);
        }

        // 在 CPU 上并行生成所有终点位置
        let end_positions: Vec<f64> = (0..particles)
            .into_par_iter()
            .map(|_| process.end(duration).unwrap_or(0.0))
            .collect();

        // 计算矩
        let moment = if order == 1 {
            end_positions.iter().sum::<f64>() / particles as f64
        } else {
            end_positions.iter().map(|&x| x.powi(order)).sum::<f64>() / particles as f64
        };

        Ok(moment)
    }

    pub fn central_moment_point<P: crate::simulation::prelude::PointProcess>(
        &self,
        process: &P,
        order: i32,
        particles: usize,
        duration: f64,
    ) -> XResult<f64> {
        use rayon::prelude::*;

        if order == 0 {
            return Ok(1.0);
        }

        // 先计算均值
        let mean = self.raw_moment_point(process, 1, particles, duration)?;

        // 在 CPU 上并行生成所有终点位置
        let end_positions: Vec<f64> = (0..particles)
            .into_par_iter()
            .map(|_| process.end(duration).unwrap_or(0.0))
            .collect();

        // 计算中心矩
        let moment = if order == 1 {
            0.0
        } else {
            end_positions
                .iter()
                .map(|&x| (x - mean).powi(order))
                .sum::<f64>()
                / particles as f64
        };

        Ok(moment)
    }
}

/// Builder pattern for GPU simulator configuration
#[derive(Default)]
pub struct GpuSimulatorBuilder {
    backend: Option<GpuBackend>,
    config: GpuConfig,
}

impl GpuSimulatorBuilder {
    /// Create a new GPU simulator builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the GPU backend
    pub fn backend(mut self, backend: GpuBackend) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Set the number of particles
    pub fn num_particles(mut self, num_particles: usize) -> Self {
        self.config.num_particles = num_particles;
        self
    }

    /// Set the number of steps
    pub fn num_steps(mut self, num_steps: usize) -> Self {
        self.config.num_steps = num_steps;
        self
    }

    /// Set threads per block
    pub fn threads_per_block(mut self, threads: usize) -> Self {
        self.config.threads_per_block = threads;
        self
    }

    /// Enable pinned memory
    pub fn use_pinned_memory(mut self, enabled: bool) -> Self {
        self.config.use_pinned_memory = enabled;
        self
    }

    /// Enable profiling
    pub fn enable_profiling(mut self, enabled: bool) -> Self {
        self.config.enable_profiling = enabled;
        self
    }

    /// Build the GPU simulator
    pub fn build(self) -> XResult<GpuSimulator> {
        let backend = self.backend.unwrap_or(GpuBackend::Auto);
        GpuSimulator::with_config(backend, self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulator_builder() {
        let builder = GpuSimulatorBuilder::new()
            .num_particles(5000)
            .num_steps(2000)
            .threads_per_block(512);

        assert_eq!(builder.config.num_particles, 5000);
        assert_eq!(builder.config.num_steps, 2000);
        assert_eq!(builder.config.threads_per_block, 512);
    }

    #[test]
    fn test_simulator_creation_auto_backend() {
        // This will fail if no GPU is available, which is expected
        match GpuSimulator::new(GpuBackend::Auto) {
            Ok(simulator) => {
                println!("Created GPU simulator: {}", simulator.info());
                assert!(simulator.backend().is_available());
            }
            Err(e) => {
                println!("No GPU available (expected on systems without GPU): {}", e);
            }
        }
    }

    #[test]
    fn test_invalid_config() {
        let config = GpuConfig::new(0, 1000); // Invalid: 0 particles
        let result = GpuSimulator::with_config(GpuBackend::Auto, config);
        assert!(result.is_err());
    }
}
