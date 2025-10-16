//! GPU traits for stochastic process implementations
//!
//! This module defines traits that stochastic processes can implement
//! to provide GPU-accelerated simulation capabilities.

use crate::{simulation::prelude::Pair, XResult};

use super::backend::GpuConfig;

/// Trait for GPU-accelerated continuous processes
///
/// Implement this trait for any continuous process that can be simulated on GPU.
/// This enables high-performance parallel simulation of many particles.
pub trait GpuContinuousProcess: Send + Sync {
    /// Simulate the process on GPU with multiple particles in parallel
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation
    /// * `time_step` - The time step for the simulation
    /// * `config` - GPU configuration including number of particles
    ///
    /// # Returns
    ///
    /// A vector of trajectories (time, position pairs), one for each particle
    fn simulate_gpu(&self, duration: f64, time_step: f64, config: &GpuConfig)
        -> XResult<Vec<Pair>>;

    /// Get the process parameters as a byte array for GPU transfer
    ///
    /// This allows efficient transfer of process-specific parameters to GPU memory.
    fn get_parameters(&self) -> Vec<f32>;

    /// Get the GPU kernel name for this process
    ///
    /// This should match the kernel function name in the GPU code.
    fn kernel_name(&self) -> &str;

    /// Check if GPU acceleration is available for this process
    fn supports_gpu(&self) -> bool {
        true
    }
}

/// Trait for GPU-accelerated moment computation
///
/// Processes implementing this trait can compute statistical moments
/// directly on GPU without transferring full trajectories back to CPU.
pub trait GpuMoments: Send + Sync {
    /// Compute raw moment on GPU
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the moment
    /// * `duration` - The duration of the simulation
    /// * `num_particles` - The number of particles to simulate
    /// * `time_step` - The time step for the simulation
    /// * `config` - GPU configuration
    ///
    /// # Returns
    ///
    /// Time series of the moment values
    fn raw_moment_gpu(
        &self,
        order: i32,
        duration: f64,
        num_particles: usize,
        time_step: f64,
        config: &GpuConfig,
    ) -> XResult<Vec<f64>>;

    /// Compute central moment on GPU
    fn central_moment_gpu(
        &self,
        order: i32,
        duration: f64,
        num_particles: usize,
        time_step: f64,
        config: &GpuConfig,
    ) -> XResult<Vec<f64>>;

    /// Compute mean square displacement on GPU
    fn msd_gpu(
        &self,
        duration: f64,
        num_particles: usize,
        time_step: f64,
        config: &GpuConfig,
    ) -> XResult<Vec<f64>> {
        self.central_moment_gpu(2, duration, num_particles, time_step, config)
    }
}

/// Trait for GPU-accelerated first passage time computation
pub trait GpuFirstPassageTime: Send + Sync {
    /// Compute first passage time statistics on GPU
    ///
    /// # Arguments
    ///
    /// * `domain` - The boundary domain (lower, upper)
    /// * `max_duration` - Maximum simulation time
    /// * `time_step` - Time step for simulation
    /// * `num_particles` - Number of particles to simulate
    /// * `config` - GPU configuration
    ///
    /// # Returns
    ///
    /// Vector of first passage times for each particle (None if didn't cross)
    fn fpt_gpu(
        &self,
        domain: (f64, f64),
        max_duration: f64,
        time_step: f64,
        num_particles: usize,
        config: &GpuConfig,
    ) -> XResult<Vec<Option<f64>>>;

    /// Compute mean first passage time on GPU
    fn mean_fpt_gpu(
        &self,
        domain: (f64, f64),
        max_duration: f64,
        time_step: f64,
        num_particles: usize,
        config: &GpuConfig,
    ) -> XResult<f64> {
        let fpts = self.fpt_gpu(domain, max_duration, time_step, num_particles, config)?;
        let sum: f64 = fpts.iter().filter_map(|&x| x).sum();
        let count = fpts.iter().filter(|x| x.is_some()).count();

        if count > 0 {
            Ok(sum / count as f64)
        } else {
            Err(crate::XError::SimulateError(
                crate::SimulationError::NoResult,
            ))
        }
    }
}

/// Trait for GPU-accelerated ensemble averaging
pub trait GpuEnsemble: Send + Sync {
    /// Compute ensemble average over many realizations on GPU
    ///
    /// # Arguments
    ///
    /// * `duration` - Duration of each realization
    /// * `time_step` - Time step for simulation
    /// * `num_realizations` - Number of independent realizations
    /// * `config` - GPU configuration
    ///
    /// # Returns
    ///
    /// Time series of ensemble-averaged values
    fn ensemble_average_gpu(
        &self,
        duration: f64,
        time_step: f64,
        num_realizations: usize,
        config: &GpuConfig,
    ) -> XResult<Vec<f64>>;
}

/// Trait for GPU random number generation specific to a process
///
/// Some processes may require specialized random number generation
/// (e.g., stable distributions, Poisson, etc.)
pub trait GpuRandomGenerator: Send + Sync {
    /// Generate random numbers on GPU for this process
    ///
    /// # Arguments
    ///
    /// * `count` - Number of random numbers to generate
    /// * `config` - GPU configuration
    ///
    /// # Returns
    ///
    /// GPU buffer containing random numbers (implementation-specific)
    fn generate_gpu_randoms(&self, count: usize, config: &GpuConfig) -> XResult<Vec<f32>>;

    /// Get the random number distribution name
    fn distribution_name(&self) -> &str;
}

/// Trait for processes that can be subordinated on GPU
pub trait GpuSubordinator: Send + Sync {
    /// Simulate subordinated process on GPU
    ///
    /// # Arguments
    ///
    /// * `parent_process` - The parent process to subordinate
    /// * `duration` - Duration of simulation
    /// * `time_step` - Time step
    /// * `config` - GPU configuration
    fn subordinate_gpu<P: GpuContinuousProcess>(
        &self,
        parent_process: &P,
        duration: f64,
        time_step: f64,
        config: &GpuConfig,
    ) -> XResult<Vec<Pair>>;
}

/// Trait for GPU memory management and optimization
pub trait GpuMemoryManaged {
    /// Estimate GPU memory required for simulation
    ///
    /// # Arguments
    ///
    /// * `num_particles` - Number of particles
    /// * `num_steps` - Number of time steps
    ///
    /// # Returns
    ///
    /// Estimated memory in bytes
    fn estimate_memory_usage(&self, num_particles: usize, num_steps: usize) -> usize {
        // Default estimation:
        // - Random numbers: num_particles * num_steps * 4 bytes (f32)
        // - Positions: num_particles * (num_steps + 1) * 4 bytes
        // - Times: num_steps * 4 bytes (shared)
        let random_mem = num_particles * num_steps * 4;
        let position_mem = num_particles * (num_steps + 1) * 4;
        let time_mem = num_steps * 4;

        random_mem + position_mem + time_mem
    }

    /// Check if simulation fits in available GPU memory
    fn fits_in_memory(
        &self,
        num_particles: usize,
        num_steps: usize,
        available_memory: usize,
    ) -> bool {
        self.estimate_memory_usage(num_particles, num_steps) <= available_memory
    }

    /// Suggest optimal batch size for memory constraints
    fn suggest_batch_size(
        &self,
        num_particles: usize,
        num_steps: usize,
        available_memory: usize,
    ) -> usize {
        let per_particle_mem = self.estimate_memory_usage(1, num_steps);

        if per_particle_mem == 0 {
            return num_particles;
        }

        let max_particles = available_memory / per_particle_mem;
        max_particles.min(num_particles)
    }
}

/// Performance profiling information for GPU operations
#[derive(Debug, Clone)]
pub struct GpuProfileInfo {
    /// Time spent on random number generation (seconds)
    pub rng_time: f64,

    /// Time spent on kernel execution (seconds)
    pub kernel_time: f64,

    /// Time spent on host-to-device transfers (seconds)
    pub h2d_transfer_time: f64,

    /// Time spent on device-to-host transfers (seconds)
    pub d2h_transfer_time: f64,

    /// Total time (seconds)
    pub total_time: f64,

    /// Number of particles simulated
    pub num_particles: usize,

    /// Number of time steps
    pub num_steps: usize,
}

impl GpuProfileInfo {
    /// Create a new profile info
    pub fn new(num_particles: usize, num_steps: usize) -> Self {
        Self {
            rng_time: 0.0,
            kernel_time: 0.0,
            h2d_transfer_time: 0.0,
            d2h_transfer_time: 0.0,
            total_time: 0.0,
            num_particles,
            num_steps,
        }
    }

    /// Get computation time (excluding transfers)
    pub fn compute_time(&self) -> f64 {
        self.rng_time + self.kernel_time
    }

    /// Get transfer time
    pub fn transfer_time(&self) -> f64 {
        self.h2d_transfer_time + self.d2h_transfer_time
    }

    /// Get compute efficiency (compute_time / total_time)
    pub fn compute_efficiency(&self) -> f64 {
        if self.total_time > 0.0 {
            self.compute_time() / self.total_time
        } else {
            0.0
        }
    }

    /// Get throughput (particles * steps / total_time)
    pub fn throughput(&self) -> f64 {
        if self.total_time > 0.0 {
            (self.num_particles * self.num_steps) as f64 / self.total_time
        } else {
            0.0
        }
    }

    /// Format as human-readable string
    pub fn to_string(&self) -> String {
        format!(
            "GPU Profile:\n\
             - RNG Time:        {:.3}s ({:.1}%)\n\
             - Kernel Time:     {:.3}s ({:.1}%)\n\
             - H2D Transfer:    {:.3}s ({:.1}%)\n\
             - D2H Transfer:    {:.3}s ({:.1}%)\n\
             - Total Time:      {:.3}s\n\
             - Particles:       {}\n\
             - Steps:           {}\n\
             - Throughput:      {:.2e} steps/s\n\
             - Compute Efficiency: {:.1}%",
            self.rng_time,
            (self.rng_time / self.total_time) * 100.0,
            self.kernel_time,
            (self.kernel_time / self.total_time) * 100.0,
            self.h2d_transfer_time,
            (self.h2d_transfer_time / self.total_time) * 100.0,
            self.d2h_transfer_time,
            (self.d2h_transfer_time / self.total_time) * 100.0,
            self.total_time,
            self.num_particles,
            self.num_steps,
            self.throughput(),
            self.compute_efficiency() * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_profile_info() {
        let mut profile = GpuProfileInfo::new(1000, 10000);
        profile.rng_time = 0.1;
        profile.kernel_time = 0.5;
        profile.h2d_transfer_time = 0.05;
        profile.d2h_transfer_time = 0.05;
        profile.total_time = 0.7;

        assert_eq!(profile.compute_time(), 0.6);
        assert_eq!(profile.transfer_time(), 0.1);
        assert!((profile.compute_efficiency() - 0.857).abs() < 0.01);
        assert!(profile.throughput() > 0.0);
    }

    #[test]
    fn test_memory_estimation() {
        struct TestProcess;
        impl GpuMemoryManaged for TestProcess {}

        let process = TestProcess;
        let mem = process.estimate_memory_usage(1000, 10000);

        // Should be roughly: 1000 * 10000 * 4 + 1000 * 10001 * 4 + 10000 * 4
        assert!(mem > 0);

        let batch = process.suggest_batch_size(10000, 1000, mem);
        assert!(batch > 0);
        assert!(batch <= 10000);
    }
}
