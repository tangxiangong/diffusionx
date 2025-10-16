//! CUDA backend implementation for GPU-accelerated stochastic process simulation
//!
//! This module provides CUDA-based implementations using NVIDIA GPUs.

use crate::{XError, XResult};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaStream, DevicePtr, DevicePtrMut, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use std::sync::Arc;

mod kernels;
pub use kernels::*;

mod rng;
pub use rng::*;

/// Check if CUDA is available on the system
pub fn is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        match CudaDevice::new(0) {
            Ok(_) => true,
            Err(_) => false,
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get the number of available CUDA devices
pub fn device_count() -> XResult<usize> {
    #[cfg(feature = "cuda")]
    {
        cudarc::driver::result::init()
            .map_err(|e| XError::GpuError(format!("Failed to initialize CUDA: {}", e)))?;

        let count = cudarc::driver::result::device::get_count()
            .map_err(|e| XError::GpuError(format!("Failed to get CUDA device count: {}", e)))?;

        Ok(count as usize)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(XError::GpuError(
            "CUDA support not enabled. Enable 'cuda' feature.".to_string(),
        ))
    }
}

/// CUDA device wrapper
#[cfg(feature = "cuda")]
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    stream: CudaStream,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Create a new CUDA backend with the specified device
    pub fn new(device_id: usize) -> XResult<Self> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| XError::GpuError(format!("Failed to create CUDA device: {}", e)))?;

        let stream = device
            .fork_default_stream()
            .map_err(|e| XError::GpuError(format!("Failed to create CUDA stream: {}", e)))?;

        Ok(Self { device, stream })
    }

    /// Get the device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get the stream
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Synchronize the device
    pub fn synchronize(&self) -> XResult<()> {
        self.device
            .synchronize()
            .map_err(|e| XError::GpuError(format!("Failed to synchronize CUDA device: {}", e)))
    }

    /// Allocate device memory
    pub fn allocate<T: cudarc::driver::DeviceRepr>(&self, len: usize) -> XResult<DevicePtrMut<T>> {
        self.device
            .alloc_zeros(len)
            .map_err(|e| XError::GpuError(format!("Failed to allocate CUDA memory: {}", e)))
    }

    /// Copy data from host to device
    pub fn copy_to_device<T: cudarc::driver::DeviceRepr>(
        &self,
        data: &[T],
    ) -> XResult<DevicePtrMut<T>> {
        self.device
            .htod_sync_copy(data)
            .map_err(|e| XError::GpuError(format!("Failed to copy to device: {}", e)))
    }

    /// Copy data from device to host
    pub fn copy_from_device<T: cudarc::driver::DeviceRepr + Clone>(
        &self,
        data: &DevicePtr<T>,
        len: usize,
    ) -> XResult<Vec<T>> {
        self.device
            .dtoh_sync_copy(data)
            .map_err(|e| XError::GpuError(format!("Failed to copy from device: {}", e)))
    }

    /// Get device information
    pub fn device_info(&self) -> XResult<CudaDeviceInfo> {
        let name = self
            .device
            .name()
            .map_err(|e| XError::GpuError(format!("Failed to get device name: {}", e)))?;

        let total_memory = self
            .device
            .total_memory()
            .map_err(|e| XError::GpuError(format!("Failed to get total memory: {}", e)))?;

        Ok(CudaDeviceInfo { name, total_memory })
    }
}

/// CUDA device information
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub name: String,
    pub total_memory: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_available() {
        let available = is_available();
        println!("CUDA available: {}", available);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_device_count() {
        match device_count() {
            Ok(count) => println!("CUDA devices: {}", count),
            Err(e) => println!("Error getting device count: {}", e),
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_backend() {
        if is_available() {
            match CudaBackend::new(0) {
                Ok(backend) => {
                    println!("Created CUDA backend");
                    if let Ok(info) = backend.device_info() {
                        println!("Device: {}", info.name);
                        println!("Memory: {} MB", info.total_memory / (1024 * 1024));
                    }
                }
                Err(e) => println!("Error creating backend: {}", e),
            }
        }
    }
}
