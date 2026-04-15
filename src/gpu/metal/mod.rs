use crate::XResult;
use metal::{Device, MTLSize};
use std::sync::LazyLock;

pub(crate) static METAL_DEVICE: LazyLock<XResult<Device>> = LazyLock::new(|| {
    Device::system_default().ok_or_else(|| crate::XError::Other("No Metal device found".into()))
});

pub(crate) static METAL_QUEUE: LazyLock<XResult<metal::CommandQueue>> = LazyLock::new(|| {
    let device = METAL_DEVICE.as_ref()?;
    Ok(device.new_command_queue())
});

// Pre-compiled Metal library paths (set by build.rs)
pub(crate) const BM_METALLIB: &str = env!("BM_KERNEL_METALLIB");
pub(crate) const LEVY_METALLIB: &str = env!("LEVY_KERNEL_METALLIB");
pub(crate) const OU_METALLIB: &str = env!("OU_KERNEL_METALLIB");
pub(crate) const RANDOM_METALLIB: &str = env!("RANDOM_KERNEL_METALLIB");

/// Load a pre-compiled Metal library from path
pub(crate) fn load_library(path: &str) -> XResult<metal::Library> {
    let device = METAL_DEVICE.as_ref()?;
    device
        .new_library_with_file(path)
        .map_err(|e| crate::XError::Other(format!("Failed to load metallib '{}': {}", path, e)))
}

/// Get compute pipeline state for a kernel function
pub(crate) fn get_pipeline(
    library: &metal::Library,
    function_name: &str,
) -> XResult<metal::ComputePipelineState> {
    let device = METAL_DEVICE.as_ref()?;
    let function = library.get_function(function_name, None).map_err(|e| {
        crate::XError::Other(format!("Function '{}' not found: {}", function_name, e))
    })?;

    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| crate::XError::Other(format!("Pipeline creation error: {}", e)))
}

/// Calculate thread group configuration for a given number of particles
#[inline]
pub(crate) fn thread_config(particles: usize) -> (MTLSize, MTLSize) {
    let thread_group_size = 256usize;
    let thread_groups = particles.div_ceil(thread_group_size);

    (
        MTLSize::new(thread_groups as u64, 1, 1),
        MTLSize::new(thread_group_size as u64, 1, 1),
    )
}

/// Macro to generate Metal GPU-accelerated moment calculation functions
macro_rules! subscribe_metal_gpu_function {
    ($library:expr, $func_name:ident, $kernel_name:expr, ($( $param_name:ident: $param_type:ty ),+ $(,)?)) => {
        #[allow(clippy::too_many_arguments)]
        fn $func_name(
            $(
                $param_name: $param_type,
            )+
            particles: usize,
        ) -> XResult<f32> {
            use metal::MTLResourceOptions;

            let device = $crate::gpu::metal::METAL_DEVICE.as_ref()?;
            let queue = $crate::gpu::metal::METAL_QUEUE.as_ref()?;
            static PIPELINE: std::sync::LazyLock<XResult<metal::ComputePipelineState>> =
                std::sync::LazyLock::new(|| {
                    let library = $library.as_ref()?;
                    $crate::gpu::metal::get_pipeline(library, $kernel_name)
                });
            let pipeline = PIPELINE.as_ref()?;

            let (thread_groups, threads_per_group) = $crate::gpu::metal::thread_config(particles);

            // Create output buffer
            let out_buffer = device.new_buffer(
                std::mem::size_of::<f32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Zero initialize output
            unsafe {
                let ptr = out_buffer.contents() as *mut f32;
                *ptr = 0.0f32;
            }

            let mut rng = rand::rng();
            use rand::RngExt;
            let seed: u64 = rng.random();
            let particles_u32 = particles as u32;

            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(pipeline);

            // Set buffers
            let mut buffer_index = 0u64;
            encoder.set_buffer(buffer_index, Some(&out_buffer), 0);
            buffer_index += 1;

            $(
                encoder.set_bytes(
                    buffer_index,
                    std::mem::size_of::<$param_type>() as u64,
                    &$param_name as *const $param_type as *const std::ffi::c_void,
                );
                buffer_index += 1;
            )+

            encoder.set_bytes(
                buffer_index,
                std::mem::size_of::<u32>() as u64,
                &particles_u32 as *const u32 as *const std::ffi::c_void,
            );
            buffer_index += 1;

            encoder.set_bytes(
                buffer_index,
                std::mem::size_of::<u64>() as u64,
                &seed as *const u64 as *const std::ffi::c_void,
            );

            // Set threadgroup memory for SIMD sums (32 floats)
            encoder.set_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>() as u64);

            encoder.dispatch_thread_groups(thread_groups, threads_per_group);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Read result
            let sum = unsafe {
                let ptr = out_buffer.contents() as *const f32;
                *ptr
            };

            Ok(sum / particles as f32)
        }
    };
}

/// Macro to generate Metal GPU-accelerated central moment calculation functions
macro_rules! subscribe_metal_central_moment_gpu_function {
    ($library:expr, $func_name:ident, $kernel_name:expr, ($( $param_name:ident: $param_type:ty ),+ $(,)?), $order_type:ty) => {
        #[allow(clippy::too_many_arguments)]
        fn $func_name(
            $(
                $param_name: $param_type,
            )+
            order: $order_type,
            particles: usize,
        ) -> XResult<f32> {
            use metal::MTLResourceOptions;

            let device = $crate::gpu::metal::METAL_DEVICE.as_ref()?;
            let queue = $crate::gpu::metal::METAL_QUEUE.as_ref()?;
            static PIPELINE: std::sync::LazyLock<XResult<metal::ComputePipelineState>> =
                std::sync::LazyLock::new(|| {
                    let library = $library.as_ref()?;
                    $crate::gpu::metal::get_pipeline(library, $kernel_name)
                });
            let pipeline = PIPELINE.as_ref()?;

            let (thread_groups, threads_per_group) = $crate::gpu::metal::thread_config(particles);

            // First compute mean
            let mean_val = mean($($param_name,)+ particles)?;

            // Create output buffer
            let out_buffer = device.new_buffer(
                std::mem::size_of::<f32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Zero initialize output
            unsafe {
                let ptr = out_buffer.contents() as *mut f32;
                *ptr = 0.0f32;
            }

            let mut rng = rand::rng();
            use rand::RngExt;
            let seed: u64 = rng.random();
            let particles_u32 = particles as u32;

            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(pipeline);

            // Set buffers - order matches kernel signature: out, order, mean, params..., particles, seed
            let mut buffer_index = 0u64;
            encoder.set_buffer(buffer_index, Some(&out_buffer), 0);
            buffer_index += 1;

            encoder.set_bytes(
                buffer_index,
                std::mem::size_of::<$order_type>() as u64,
                &order as *const $order_type as *const std::ffi::c_void,
            );
            buffer_index += 1;

            encoder.set_bytes(
                buffer_index,
                std::mem::size_of::<f32>() as u64,
                &mean_val as *const f32 as *const std::ffi::c_void,
            );
            buffer_index += 1;

            $(
                encoder.set_bytes(
                    buffer_index,
                    std::mem::size_of::<$param_type>() as u64,
                    &$param_name as *const $param_type as *const std::ffi::c_void,
                );
                buffer_index += 1;
            )+

            encoder.set_bytes(
                buffer_index,
                std::mem::size_of::<u32>() as u64,
                &particles_u32 as *const u32 as *const std::ffi::c_void,
            );
            buffer_index += 1;

            encoder.set_bytes(
                buffer_index,
                std::mem::size_of::<u64>() as u64,
                &seed as *const u64 as *const std::ffi::c_void,
            );

            // Set threadgroup memory for SIMD sums (32 floats)
            encoder.set_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>() as u64);

            encoder.dispatch_thread_groups(thread_groups, threads_per_group);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Read result
            let sum = unsafe {
                let ptr = out_buffer.contents() as *const f32;
                *ptr
            };

            Ok(sum / particles as f32)
        }
    };
}

pub mod bm;
pub mod levy;
pub mod ou;
pub mod random;
