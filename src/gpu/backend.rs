//! GPU backend trait for unified interface
//!
//! This module defines the common interface that all GPU backends must implement,
//! allowing for backend-agnostic GPU-accelerated simulation code.

use crate::XResult;

/// Trait for GPU backend implementations
pub trait GpuBackendTrait: Send + Sync {
    /// Get the backend name
    fn name(&self) -> &str;

    /// Check if the backend is available
    fn is_available(&self) -> bool;

    /// Get backend information as a string
    fn info(&self) -> String;

    /// Synchronize the GPU (wait for all operations to complete)
    fn synchronize(&self) -> XResult<()>;

    /// Allocate GPU memory
    fn allocate_memory(&self, size_bytes: usize) -> XResult<Box<dyn GpuBuffer>>;

    /// Get maximum number of threads per block/threadgroup
    fn max_threads_per_block(&self) -> usize;

    /// Get total GPU memory in bytes
    fn total_memory(&self) -> XResult<usize>;

    /// Get available (free) GPU memory in bytes
    fn available_memory(&self) -> XResult<usize>;
}

/// Trait for GPU memory buffers
pub trait GpuBuffer: Send + Sync {
    /// Get the size of the buffer in bytes
    fn size(&self) -> usize;

    /// Copy bytes from host to device
    fn copy_bytes_from_host(&mut self, data: &[u8]) -> XResult<()>;

    /// Copy bytes from device to host
    fn copy_bytes_to_host(&self, data: &mut [u8]) -> XResult<()>;

    /// Get a pointer identifier (for debugging/logging)
    fn ptr_id(&self) -> usize;
}

/// Configuration for GPU simulation
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Number of particles to simulate in parallel
    pub num_particles: usize,

    /// Number of time steps
    pub num_steps: usize,

    /// Threads per block (CUDA) or threadgroup (Metal)
    pub threads_per_block: usize,

    /// Whether to use pinned memory for faster transfers
    pub use_pinned_memory: bool,

    /// Whether to enable profiling
    pub enable_profiling: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            num_particles: 1000,
            num_steps: 1000,
            threads_per_block: 256,
            use_pinned_memory: false,
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

    /// Enable pinned memory
    pub fn with_pinned_memory(mut self, enabled: bool) -> Self {
        self.use_pinned_memory = enabled;
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

    /// Calculate total number of random numbers needed
    pub fn total_randoms_needed(&self) -> usize {
        self.num_particles * self.num_steps
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

/// GPU memory statistics
#[derive(Debug, Clone)]
pub struct GpuMemoryStats {
    /// Total memory in bytes
    pub total: usize,

    /// Used memory in bytes
    pub used: usize,

    /// Free memory in bytes
    pub free: usize,
}

impl GpuMemoryStats {
    /// Get memory usage as a percentage
    pub fn usage_percent(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.used as f64 / self.total as f64) * 100.0
        }
    }

    /// Check if memory is critically low (>90% used)
    pub fn is_critical(&self) -> bool {
        self.usage_percent() > 90.0
    }

    /// Get a human-readable string representation
    pub fn to_string_humanized(&self) -> String {
        format!(
            "{:.2} MB / {:.2} MB ({:.1}% used)",
            self.used as f64 / (1024.0 * 1024.0),
            self.total as f64 / (1024.0 * 1024.0),
            self.usage_percent()
        )
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
            .with_pinned_memory(true)
            .with_profiling(true);

        assert_eq!(config.num_particles, 5000);
        assert_eq!(config.num_steps, 2000);
        assert_eq!(config.threads_per_block, 512);
        assert!(config.use_pinned_memory);
        assert!(config.enable_profiling);
    }

    #[test]
    fn test_num_blocks_calculation() {
        let config = GpuConfig::new(1000, 1000).with_threads_per_block(256);
        assert_eq!(config.num_blocks(), 4); // ceil(1000 / 256) = 4

        let config = GpuConfig::new(512, 1000).with_threads_per_block(256);
        assert_eq!(config.num_blocks(), 2); // ceil(512 / 256) = 2
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
    fn test_memory_stats() {
        let stats = GpuMemoryStats {
            total: 8_000_000_000,
            used: 2_000_000_000,
            free: 6_000_000_000,
        };

        assert_eq!(stats.usage_percent(), 25.0);
        assert!(!stats.is_critical());

        let stats = GpuMemoryStats {
            total: 8_000_000_000,
            used: 7_500_000_000,
            free: 500_000_000,
        };

        assert!(stats.usage_percent() > 90.0);
        assert!(stats.is_critical());
    }
}
