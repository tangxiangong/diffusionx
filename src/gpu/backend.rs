//! GPU backend configuration and types
//!
//! This module provides configuration types for GPU-accelerated simulation.

use crate::XResult;

/// Configuration for GPU simulation
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Number of particles to simulate in parallel
    pub num_particles: usize,

    /// Number of time steps
    pub num_steps: usize,

    /// Threads per block (CUDA) or threadgroup (Metal)
    pub threads_per_block: usize,

    /// Whether to enable profiling
    pub enable_profiling: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            num_particles: 1000,
            num_steps: 1000,
            threads_per_block: 256,
            enable_profiling: false,
        }
    }
}

impl GpuConfig {
    /// Create a new GPU configuration
    pub fn new(num_particles: usize, num_steps: usize) -> Self {
        Self {
            num_particles,
            num_steps,
            ..Default::default()
        }
    }

    /// Set threads per block
    pub fn with_threads_per_block(mut self, threads: usize) -> Self {
        self.threads_per_block = threads;
        self
    }

    /// Enable profiling
    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.enable_profiling = enabled;
        self
    }

    /// Calculate the number of blocks needed
    pub fn num_blocks(&self) -> usize {
        (self.num_particles + self.threads_per_block - 1) / self.threads_per_block
    }

    /// Validate the configuration
    pub fn validate(&self) -> XResult<()> {
        if self.num_particles == 0 {
            return Err(crate::XError::InvalidParameters(
                "num_particles must be greater than 0".to_string(),
            ));
        }

        if self.num_steps == 0 {
            return Err(crate::XError::InvalidParameters(
                "num_steps must be greater than 0".to_string(),
            ));
        }

        if self.threads_per_block == 0 {
            return Err(crate::XError::InvalidParameters(
                "threads_per_block must be greater than 0".to_string(),
            ));
        }

        if self.threads_per_block > 1024 {
            return Err(crate::XError::InvalidParameters(
                "threads_per_block should not exceed 1024".to_string(),
            ));
        }

        Ok(())
    }
}

/// GPU backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// CUDA backend (NVIDIA GPUs)
    Cuda,
    /// Metal backend (Apple GPUs)
    Metal,
    /// Auto-detect best available backend
    Auto,
}

impl GpuBackend {
    /// Detect the best available GPU backend
    pub fn detect() -> XResult<Self> {
        #[cfg(feature = "cuda")]
        {
            if crate::gpu::cuda::is_available() {
                return Ok(Self::Cuda);
            }
        }

        #[cfg(feature = "metal")]
        {
            if crate::gpu::metal::is_available() {
                return Ok(Self::Metal);
            }
        }

        Err(crate::XError::GpuError(
            "No GPU backend available. Enable 'cuda' or 'metal' feature.".to_string(),
        ))
    }

    /// Check if this backend is available
    pub fn is_available(&self) -> bool {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda => crate::gpu::cuda::is_available(),
            #[cfg(not(feature = "cuda"))]
            Self::Cuda => false,

            #[cfg(feature = "metal")]
            Self::Metal => crate::gpu::metal::is_available(),
            #[cfg(not(feature = "metal"))]
            Self::Metal => false,

            Self::Auto => Self::detect().is_ok(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.num_particles, 1000);
        assert_eq!(config.num_steps, 1000);
        assert_eq!(config.threads_per_block, 256);
    }

    #[test]
    fn test_gpu_config_builder() {
        let config = GpuConfig::new(5000, 2000)
            .with_threads_per_block(512)
            .with_profiling(true);

        assert_eq!(config.num_particles, 5000);
        assert_eq!(config.num_steps, 2000);
        assert_eq!(config.threads_per_block, 512);
        assert!(config.enable_profiling);
    }

    #[test]
    fn test_num_blocks_calculation() {
        let config = GpuConfig::new(1000, 1000).with_threads_per_block(256);
        assert_eq!(config.num_blocks(), 4);

        let config = GpuConfig::new(512, 1000).with_threads_per_block(256);
        assert_eq!(config.num_blocks(), 2);
    }

    #[test]
    fn test_validate() {
        let config = GpuConfig::new(1000, 1000);
        assert!(config.validate().is_ok());

        let config = GpuConfig::new(0, 1000);
        assert!(config.validate().is_err());

        let config = GpuConfig::new(1000, 0);
        assert!(config.validate().is_err());

        let config = GpuConfig::new(1000, 1000).with_threads_per_block(0);
        assert!(config.validate().is_err());

        let config = GpuConfig::new(1000, 1000).with_threads_per_block(2048);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_backend_detect() {
        let _result = GpuBackend::detect();
        // Result depends on system, just make sure it doesn't panic
    }
}
