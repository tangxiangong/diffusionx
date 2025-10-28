//! GPU traits and types for stochastic process simulation
//!
//! This module provides minimal trait definitions for GPU acceleration.

/// Performance profiling information for GPU operations
#[derive(Debug, Clone)]
pub struct GpuProfileInfo {
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
            kernel_time: 0.0,
            h2d_transfer_time: 0.0,
            d2h_transfer_time: 0.0,
            total_time: 0.0,
            num_particles,
            num_steps,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_profile_info() {
        let mut profile = GpuProfileInfo::new(1000, 10000);
        profile.kernel_time = 0.5;
        profile.h2d_transfer_time = 0.05;
        profile.d2h_transfer_time = 0.05;
        profile.total_time = 0.6;

        assert!(profile.throughput() > 0.0);
    }
}
