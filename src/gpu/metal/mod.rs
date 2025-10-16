//! Metal backend implementation for GPU-accelerated stochastic process simulation
//!
//! This module provides Metal-based implementations using Apple GPUs (macOS, iOS).

use crate::{XError, XResult};

#[cfg(feature = "metal")]
use metal::*;
#[cfg(feature = "metal")]
mod kernels;
pub use kernels::*;

mod rng;
pub use rng::*;

/// Check if Metal is available on the system
pub fn is_available() -> bool {
    #[cfg(feature = "metal")]
    {
        Device::all().len() > 0
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
}

/// Get the number of available Metal devices
pub fn device_count() -> XResult<usize> {
    #[cfg(feature = "metal")]
    {
        Ok(Device::all().len())
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(XError::GpuError(
            "Metal support not enabled. Enable 'metal' feature.".to_string(),
        ))
    }
}

/// Metal device wrapper
#[cfg(feature = "metal")]
pub struct MetalBackend {
    device: Device,
    command_queue: CommandQueue,
    library: Library,
}

#[cfg(feature = "metal")]
impl MetalBackend {
    /// Create a new Metal backend with the default device
    pub fn new() -> XResult<Self> {
        let device = Device::system_default()
            .ok_or_else(|| XError::GpuError("No Metal device found".to_string()))?;

        let command_queue = device.new_command_queue();

        // Compile the Metal shaders
        let library = Self::compile_library(&device)?;

        Ok(Self {
            device,
            command_queue,
            library,
        })
    }

    /// Create a Metal backend with a specific device
    pub fn with_device(device_id: usize) -> XResult<Self> {
        let devices = Device::all();
        let device = devices
            .get(device_id)
            .ok_or_else(|| {
                XError::GpuError(format!(
                    "Metal device {} not found. Available devices: {}",
                    device_id,
                    devices.len()
                ))
            })?
            .clone();

        let command_queue = device.new_command_queue();
        let library = Self::compile_library(&device)?;

        Ok(Self {
            device,
            command_queue,
            library,
        })
    }

    /// Compile Metal shader library
    fn compile_library(device: &Device) -> XResult<Library> {
        let source = kernels::METAL_SHADERS;
        let compile_options = CompileOptions::new();

        device
            .new_library_with_source(source, &compile_options)
            .map_err(|e| XError::GpuError(format!("Failed to compile Metal shaders: {}", e)))
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the command queue
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    /// Get the shader library
    pub fn library(&self) -> &Library {
        &self.library
    }

    /// Create a buffer with data
    pub fn create_buffer<T>(&self, data: &[T]) -> XResult<Buffer> {
        let size = std::mem::size_of_val(data) as u64;
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            size,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(buffer)
    }

    /// Create an empty buffer
    pub fn create_empty_buffer(&self, size: u64) -> XResult<Buffer> {
        let buffer = self
            .device
            .new_buffer(size, MTLResourceOptions::StorageModeShared);
        Ok(buffer)
    }

    /// Read data from buffer
    pub fn read_buffer<T: Clone>(&self, buffer: &Buffer, count: usize) -> XResult<Vec<T>> {
        let ptr = buffer.contents() as *const T;
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
        Ok(slice.to_vec())
    }

    /// Get device information
    pub fn device_info(&self) -> MetalDeviceInfo {
        MetalDeviceInfo {
            name: self.device.name().to_string(),
            is_low_power: self.device.is_low_power(),
            is_headless: self.device.is_headless(),
            recommended_max_working_set_size: self.device.recommended_max_working_set_size(),
            max_threads_per_threadgroup: self.device.max_threads_per_threadgroup(),
        }
    }

    /// Create a command buffer
    pub fn create_command_buffer(&self) -> &CommandBufferRef {
        self.command_queue.new_command_buffer()
    }

    /// Execute a compute command
    pub fn execute_compute<F>(&self, setup: F) -> XResult<()>
    where
        F: FnOnce(&ComputeCommandEncoderRef) -> XResult<()>,
    {
        let command_buffer = self.create_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        setup(encoder)?;

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}

/// Metal device information
#[derive(Debug, Clone)]
pub struct MetalDeviceInfo {
    pub name: String,
    pub is_low_power: bool,
    pub is_headless: bool,
    pub recommended_max_working_set_size: u64,
    pub max_threads_per_threadgroup: MTLSize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_available() {
        let available = is_available();
        println!("Metal available: {}", available);
    }

    #[test]
    #[cfg(feature = "metal")]
    fn test_device_count() {
        match device_count() {
            Ok(count) => println!("Metal devices: {}", count),
            Err(e) => println!("Error getting device count: {}", e),
        }
    }

    #[test]
    #[cfg(feature = "metal")]
    fn test_metal_backend() {
        if is_available() {
            match MetalBackend::new() {
                Ok(backend) => {
                    println!("Created Metal backend");
                    let info = backend.device_info();
                    println!("Device: {}", info.name);
                    println!("Low Power: {}", info.is_low_power);
                    println!(
                        "Max Working Set: {} MB",
                        info.recommended_max_working_set_size / (1024 * 1024)
                    );
                    println!(
                        "Max Threads Per Threadgroup: {}x{}x{}",
                        info.max_threads_per_threadgroup.width,
                        info.max_threads_per_threadgroup.height,
                        info.max_threads_per_threadgroup.depth
                    );
                }
                Err(e) => println!("Error creating backend: {}", e),
            }
        }
    }
}
