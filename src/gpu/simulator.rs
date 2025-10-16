//! GPU simulator for stochastic process simulation
//!
//! This module provides a high-level interface for GPU-accelerated simulation
//! of stochastic processes. It abstracts away backend-specific details and
//! provides a unified API for both CUDA and Metal.

use crate::{
    XResult,
    simulation::continuous::{Bm, GeometricBm, OrnsteinUhlenbeck},
    simulation::prelude::{ContinuousProcess, Pair},
    utils::linspace,
};

use super::{GpuBackend, backend::GpuConfig, montecarlo::MonteCarloStats};

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

    /// Simulate a stochastic process on GPU with multiple particles
    ///
    /// # Arguments
    ///
    /// * `process` - The stochastic process to simulate
    /// * `duration` - The duration of the simulation
    /// * `time_step` - The time step for the simulation
    /// * `num_particles` - The number of particles to simulate in parallel
    ///
    /// # Returns
    ///
    /// A vector of trajectories, one for each particle
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use diffusionx::gpu::{GpuSimulator, GpuBackend};
    /// use diffusionx::simulation::continuous::Bm;
    ///
    /// let simulator = GpuSimulator::new(GpuBackend::Auto)?;
    /// let bm = Bm::default();
    ///
    /// // Simulate 1000 particles in parallel on GPU
    /// let trajectories = simulator.simulate_parallel(&bm, 1.0, 0.01, 1000)?;
    /// ```
    pub fn simulate_parallel<P: ContinuousProcess>(
        &self,
        _process: &P,
        _duration: f64,
        _time_step: f64,
        _num_particles: usize,
    ) -> XResult<Vec<Pair>> {
        // This is a placeholder for the actual GPU implementation
        // The actual implementation would:
        // 1. Allocate GPU memory for random numbers and results
        // 2. Generate random numbers on GPU or transfer from CPU
        // 3. Launch appropriate kernel based on process type
        // 4. Copy results back to CPU
        // 5. Return trajectories

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

        let kernel = library
            .get_function("simulate_bm", None)
            .map_err(|e| crate::XError::GpuError(format!("Failed to get kernel: {}", e)))?;

        let rng = MetalRng::new(device.clone());
        let randoms_buffer = rng.standard_normals_f32(num_particles * num_steps)?;

        let output_size = num_particles * (num_steps + 1);
        let positions_size = (output_size * std::mem::size_of::<f32>()) as u64;
        let positions_buffer = metal.create_empty_buffer(positions_size)?;

        let command_queue = metal.command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(&kernel);
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
        let num_groups = (num_particles + threads_per_group - 1) / threads_per_group;

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

    /// Get GPU information string
    pub fn info(&self) -> String {
        format!("GPU Simulator with backend: {:?}", self.backend)
    }
}

/// Builder pattern for GPU simulator configuration
pub struct GpuSimulatorBuilder {
    backend: Option<GpuBackend>,
    config: GpuConfig,
}

impl Default for GpuSimulatorBuilder {
    fn default() -> Self {
        Self {
            backend: None,
            config: GpuConfig::default(),
        }
    }
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
