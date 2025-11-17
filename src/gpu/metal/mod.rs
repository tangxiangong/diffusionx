//! Metal backend implementation for GPU-accelerated stochastic process simulation
//!
//! This module provides Metal-based implementations using Apple GPUs (macOS, iOS).

use crate::{XError, XResult};

#[cfg(feature = "metal")]
use metal::*;

mod rng;
pub use rng::*;

/// Check if Metal is available on the system
pub fn is_available() -> bool {
    #[cfg(feature = "metal")]
    {
        !Device::all().is_empty()
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

    /// Compile Metal shader library from embedded source code
    fn compile_library(device: &Device) -> XResult<Library> {
        // Get all Metal shader sources from the metal-kernel crate
        let combined_source = metal_kernel::all_shaders();

        let compile_options = CompileOptions::new();
        device
            .new_library_with_source(&combined_source, &compile_options)
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

    /// Generate Stable distributed random numbers on Metal GPU
    pub fn generate_stable(&self, n: usize, alpha: f32, beta: f32, seed: u32) -> XResult<Vec<f32>> {
        // 获取内核函数
        let kernel = self
            .library
            .get_function("generate_stable", None)
            .map_err(|e| XError::GpuError(format!("Failed to get kernel: {}", e)))?;

        let pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| XError::GpuError(format!("Failed to create pipeline: {}", e)))?;

        // 创建输出缓冲区
        let output_size = (n * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.create_empty_buffer(output_size)?;

        // 创建命令缓冲区和编码器
        let command_buffer = self.create_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&output_buffer), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of::<f32>() as u64,
            &alpha as *const f32 as *const _,
        );
        encoder.set_bytes(
            2,
            std::mem::size_of::<f32>() as u64,
            &beta as *const f32 as *const _,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &seed as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &(n as u32) as *const u32 as *const _,
        );

        // 设置线程组大小
        let threads_per_group = 256;
        let num_groups = n.div_ceil(threads_per_group);

        encoder.dispatch_thread_groups(
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

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // 读取结果
        self.read_buffer(&output_buffer, n)
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
