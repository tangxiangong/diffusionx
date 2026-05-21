use crate::{XError, XResult};
use dispatch2::DispatchData;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLFunction, MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::{ffi::c_void, ptr::NonNull, sync::LazyLock};

// `MTLCreateSystemDefaultDevice` returns `nil` unless CoreGraphics is linked into
// the process. No symbol from CoreGraphics is referenced directly, so force the
// link with an empty `extern` block; removing it breaks headless device creation.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

pub(crate) type MetalDevice = ProtocolObject<dyn MTLDevice>;
pub(crate) type MetalQueue = ProtocolObject<dyn MTLCommandQueue>;
pub(crate) type MetalLibrary = ProtocolObject<dyn MTLLibrary>;
pub(crate) type MetalFunction = ProtocolObject<dyn MTLFunction>;
pub(crate) type MetalPipeline = ProtocolObject<dyn MTLComputePipelineState>;
pub(crate) type MetalBuffer = ProtocolObject<dyn MTLBuffer>;
pub(crate) type MetalCommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;
pub(crate) type MetalComputeEncoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

pub(crate) static METAL_DEVICE: LazyLock<XResult<Retained<MetalDevice>>> = LazyLock::new(|| {
    MTLCreateSystemDefaultDevice().ok_or_else(|| XError::Other("No Metal device found".into()))
});

pub(crate) static METAL_QUEUE: LazyLock<XResult<Retained<MetalQueue>>> = LazyLock::new(|| {
    let device = METAL_DEVICE.as_ref().map_err(Clone::clone)?;
    device
        .newCommandQueue()
        .ok_or_else(|| XError::Other("Failed to create Metal command queue".into()))
});

// Pre-compiled Metal libraries, embedded into the binary at build time.
// build.rs compiles each `.metal` source to a `.metallib` and exports its path;
// `include_bytes!` then bakes the bytes in, so the binary is self-contained and
// does not depend on the build directory existing at runtime.
pub(crate) const BM_METALLIB: &[u8] = include_bytes!(env!("BM_KERNEL_METALLIB"));
pub(crate) const LEVY_METALLIB: &[u8] = include_bytes!(env!("LEVY_KERNEL_METALLIB"));
pub(crate) const OU_METALLIB: &[u8] = include_bytes!(env!("OU_KERNEL_METALLIB"));
pub(crate) const RANDOM_METALLIB: &[u8] = include_bytes!(env!("RANDOM_KERNEL_METALLIB"));

/// Load a pre-compiled Metal library from bytes embedded in the binary.
pub(crate) fn load_library(metallib: &'static [u8]) -> XResult<Retained<MetalLibrary>> {
    let device = METAL_DEVICE.as_ref().map_err(Clone::clone)?;
    let data = DispatchData::from_static_bytes(metallib);

    device
        .newLibraryWithData_error(&data)
        .map_err(|e| XError::Other(format!("Failed to load embedded metallib: {e}")))
}

/// Get compute pipeline state for a kernel function
pub(crate) fn get_pipeline(
    library: &MetalLibrary,
    function_name: &str,
) -> XResult<Retained<MetalPipeline>> {
    let device = METAL_DEVICE.as_ref().map_err(Clone::clone)?;
    let function_name_ns = NSString::from_str(function_name);
    let function: Retained<MetalFunction> = library
        .newFunctionWithName(&function_name_ns)
        .ok_or_else(|| XError::Other(format!("Function '{}' not found", function_name)))?;

    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| {
            XError::Other(format!(
                "Pipeline creation error for '{function_name}': {e}"
            ))
        })
}

pub(crate) fn new_shared_buffer(bytes: usize) -> XResult<Retained<MetalBuffer>> {
    let device = METAL_DEVICE.as_ref().map_err(Clone::clone)?;
    device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| XError::Other(format!("Failed to allocate Metal buffer ({bytes} bytes)")))
}

pub(crate) fn zero_buffer_f32(buffer: &MetalBuffer) {
    // SAFETY: `contents` is valid for a shared buffer allocated by this module, and
    // the caller passes a buffer with space for at least one `f32`.
    unsafe {
        *buffer.contents().as_ptr().cast::<f32>() = 0.0;
    }
}

pub(crate) fn read_buffer_f32(buffer: &MetalBuffer) -> f32 {
    // SAFETY: command completion is checked before reads, and moment buffers hold
    // at least one `f32` in shared storage.
    unsafe { *buffer.contents().as_ptr().cast::<f32>() }
}

pub(crate) fn read_buffer_vec_f32(buffer: &MetalBuffer, len: usize) -> Vec<f32> {
    // SAFETY: command completion is checked before reads, and callers provide the
    // number of `f32` values used when the shared buffer was allocated.
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<f32>(), len).to_vec() }
}

pub(crate) fn set_buffer(encoder: &MetalComputeEncoder, index: usize, buffer: &MetalBuffer) {
    // SAFETY: the buffer is retained by the caller until command completion, offset
    // is zero, and binding indices match the compiled Metal kernels.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
    }
}

pub(crate) fn set_scalar<T: Copy>(encoder: &MetalComputeEncoder, index: usize, value: &T) {
    let ptr = NonNull::from(value).cast::<c_void>();
    // SAFETY: `value` is a valid pointer for `size_of::<T>()` bytes; Metal copies
    // scalar bytes into the encoder immediately for the requested binding index.
    unsafe {
        encoder.setBytes_length_atIndex(ptr, std::mem::size_of::<T>(), index);
    }
}

pub(crate) fn set_threadgroup_memory_length(
    encoder: &MetalComputeEncoder,
    index: usize,
    bytes: usize,
) {
    // SAFETY: the binding index and byte length mirror the existing kernels'
    // threadgroup scratch contract.
    unsafe {
        encoder.setThreadgroupMemoryLength_atIndex(bytes, index);
    }
}

pub(crate) fn new_command_buffer(queue: &MetalQueue) -> XResult<Retained<MetalCommandBuffer>> {
    queue
        .commandBuffer()
        .ok_or_else(|| XError::Other("Failed to create Metal command buffer".into()))
}

pub(crate) fn new_compute_encoder(
    command_buffer: &MetalCommandBuffer,
) -> XResult<Retained<MetalComputeEncoder>> {
    command_buffer
        .computeCommandEncoder()
        .ok_or_else(|| XError::Other("Failed to create Metal compute encoder".into()))
}

pub(crate) fn set_pipeline(encoder: &MetalComputeEncoder, pipeline: &MetalPipeline) {
    encoder.setComputePipelineState(pipeline);
}

pub(crate) fn dispatch_threadgroups(
    encoder: &MetalComputeEncoder,
    thread_groups: MTLSize,
    threads_per_group: MTLSize,
) {
    encoder.dispatchThreadgroups_threadsPerThreadgroup(thread_groups, threads_per_group);
}

pub(crate) fn end_encoding(encoder: &MetalComputeEncoder) {
    encoder.endEncoding();
}

pub(crate) fn finish_command_buffer(command_buffer: &MetalCommandBuffer) -> XResult<()> {
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    if command_buffer.status() == MTLCommandBufferStatus::Error {
        let message = command_buffer.error().map_or_else(
            || "Metal command buffer failed".to_string(),
            |error| error.localizedDescription().to_string(),
        );
        return Err(XError::Other(message));
    }

    Ok(())
}

/// Calculate thread group configuration for a given number of particles
#[inline]
pub(crate) fn thread_config(particles: usize) -> (MTLSize, MTLSize) {
    let thread_group_size = 256usize;
    let thread_groups = particles.div_ceil(thread_group_size);

    (
        MTLSize {
            width: thread_groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: thread_group_size,
            height: 1,
            depth: 1,
        },
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
            let queue = $crate::gpu::metal::METAL_QUEUE.as_ref().map_err(Clone::clone)?;
            static PIPELINE: std::sync::LazyLock<XResult<objc2::rc::Retained<$crate::gpu::metal::MetalPipeline>>> =
                std::sync::LazyLock::new(|| {
                    let library = $library.as_ref().map_err(Clone::clone)?;
                    $crate::gpu::metal::get_pipeline(library, $kernel_name)
                });
            let pipeline = PIPELINE.as_ref().map_err(Clone::clone)?;

            let (thread_groups, threads_per_group) = $crate::gpu::metal::thread_config(particles);

            let out_buffer =
                $crate::gpu::metal::new_shared_buffer(std::mem::size_of::<f32>())?;
            $crate::gpu::metal::zero_buffer_f32(&out_buffer);

            let mut rng = rand::rng();
            use rand::RngExt;
            let seed: u64 = rng.random();
            let particles_u32 = particles as u32;

            let command_buffer = $crate::gpu::metal::new_command_buffer(queue)?;
            let encoder = $crate::gpu::metal::new_compute_encoder(&command_buffer)?;

            $crate::gpu::metal::set_pipeline(&encoder, pipeline);

            let mut buffer_index = 0usize;
            $crate::gpu::metal::set_buffer(&encoder, buffer_index, &out_buffer);
            buffer_index += 1;

            $(
                $crate::gpu::metal::set_scalar(&encoder, buffer_index, &$param_name);
                buffer_index += 1;
            )+

            $crate::gpu::metal::set_scalar(&encoder, buffer_index, &particles_u32);
            buffer_index += 1;

            $crate::gpu::metal::set_scalar(&encoder, buffer_index, &seed);

            $crate::gpu::metal::set_threadgroup_memory_length(
                &encoder,
                0,
                32 * std::mem::size_of::<f32>(),
            );

            $crate::gpu::metal::dispatch_threadgroups(&encoder, thread_groups, threads_per_group);
            $crate::gpu::metal::end_encoding(&encoder);

            $crate::gpu::metal::finish_command_buffer(&command_buffer)?;

            let sum = $crate::gpu::metal::read_buffer_f32(&out_buffer);

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
            let queue = $crate::gpu::metal::METAL_QUEUE.as_ref().map_err(Clone::clone)?;
            static PIPELINE: std::sync::LazyLock<XResult<objc2::rc::Retained<$crate::gpu::metal::MetalPipeline>>> =
                std::sync::LazyLock::new(|| {
                    let library = $library.as_ref().map_err(Clone::clone)?;
                    $crate::gpu::metal::get_pipeline(library, $kernel_name)
                });
            let pipeline = PIPELINE.as_ref().map_err(Clone::clone)?;

            let (thread_groups, threads_per_group) = $crate::gpu::metal::thread_config(particles);

            // First compute mean
            let mean_val = mean($($param_name,)+ particles)?;

            let out_buffer =
                $crate::gpu::metal::new_shared_buffer(std::mem::size_of::<f32>())?;
            $crate::gpu::metal::zero_buffer_f32(&out_buffer);

            let mut rng = rand::rng();
            use rand::RngExt;
            let seed: u64 = rng.random();
            let particles_u32 = particles as u32;

            let command_buffer = $crate::gpu::metal::new_command_buffer(queue)?;
            let encoder = $crate::gpu::metal::new_compute_encoder(&command_buffer)?;

            $crate::gpu::metal::set_pipeline(&encoder, pipeline);

            // Set buffers - order matches kernel signature: out, order, mean, params..., particles, seed
            let mut buffer_index = 0usize;
            $crate::gpu::metal::set_buffer(&encoder, buffer_index, &out_buffer);
            buffer_index += 1;

            $crate::gpu::metal::set_scalar(&encoder, buffer_index, &order);
            buffer_index += 1;

            $crate::gpu::metal::set_scalar(&encoder, buffer_index, &mean_val);
            buffer_index += 1;

            $(
                $crate::gpu::metal::set_scalar(&encoder, buffer_index, &$param_name);
                buffer_index += 1;
            )+

            $crate::gpu::metal::set_scalar(&encoder, buffer_index, &particles_u32);
            buffer_index += 1;

            $crate::gpu::metal::set_scalar(&encoder, buffer_index, &seed);

            $crate::gpu::metal::set_threadgroup_memory_length(
                &encoder,
                0,
                32 * std::mem::size_of::<f32>(),
            );

            $crate::gpu::metal::dispatch_threadgroups(&encoder, thread_groups, threads_per_group);
            $crate::gpu::metal::end_encoding(&encoder);

            $crate::gpu::metal::finish_command_buffer(&command_buffer)?;

            let sum = $crate::gpu::metal::read_buffer_f32(&out_buffer);

            Ok(sum / particles as f32)
        }
    };
}

/// Macro to generate Metal GPU-accelerated fractional central moment functions.
macro_rules! subscribe_metal_frac_central_moment_gpu_function {
    (
        $library:expr,
        $func_name:ident,
        $kernel_name:expr,
        ($( $param_name:ident: $param_type:ty ),+ $(,)?),
        before_order = ($( $before_order:ident ),* $(,)?),
        after_order = ($( $after_order:ident ),* $(,)?)
    ) => {
        #[allow(clippy::too_many_arguments)]
        fn $func_name(
            $(
                $param_name: $param_type,
            )+
            order: f32,
            particles: usize,
        ) -> XResult<f32> {
            let queue = $crate::gpu::metal::METAL_QUEUE.as_ref().map_err(Clone::clone)?;
            static PIPELINE: std::sync::LazyLock<XResult<objc2::rc::Retained<$crate::gpu::metal::MetalPipeline>>> =
                std::sync::LazyLock::new(|| {
                    let library = $library.as_ref().map_err(Clone::clone)?;
                    $crate::gpu::metal::get_pipeline(library, $kernel_name)
                });
            let pipeline = PIPELINE.as_ref().map_err(Clone::clone)?;

            let (thread_groups, threads_per_group) = $crate::gpu::metal::thread_config(particles);

            let mean_val = mean($($param_name,)+ particles)?;

            let out_buffer =
                $crate::gpu::metal::new_shared_buffer(std::mem::size_of::<f32>())?;
            $crate::gpu::metal::zero_buffer_f32(&out_buffer);

            let mut rng = rand::rng();
            use rand::RngExt;
            let seed: u64 = rng.random();
            let particles_u32 = particles as u32;

            let command_buffer = $crate::gpu::metal::new_command_buffer(queue)?;
            let encoder = $crate::gpu::metal::new_compute_encoder(&command_buffer)?;

            $crate::gpu::metal::set_pipeline(&encoder, pipeline);

            let mut buffer_index = 0usize;
            $crate::gpu::metal::set_buffer(&encoder, buffer_index, &out_buffer);
            buffer_index += 1;

            $crate::gpu::metal::set_scalar(&encoder, buffer_index, &mean_val);
            buffer_index += 1;

            $(
                $crate::gpu::metal::set_scalar(&encoder, buffer_index, &$before_order);
                buffer_index += 1;
            )*

            $crate::gpu::metal::set_scalar(&encoder, buffer_index, &order);
            buffer_index += 1;

            $(
                $crate::gpu::metal::set_scalar(&encoder, buffer_index, &$after_order);
                buffer_index += 1;
            )*

            $crate::gpu::metal::set_scalar(&encoder, buffer_index, &particles_u32);
            buffer_index += 1;

            $crate::gpu::metal::set_scalar(&encoder, buffer_index, &seed);

            $crate::gpu::metal::set_threadgroup_memory_length(
                &encoder,
                0,
                32 * std::mem::size_of::<f32>(),
            );

            $crate::gpu::metal::dispatch_threadgroups(&encoder, thread_groups, threads_per_group);
            $crate::gpu::metal::end_encoding(&encoder);

            $crate::gpu::metal::finish_command_buffer(&command_buffer)?;

            let sum = $crate::gpu::metal::read_buffer_f32(&out_buffer);

            Ok(sum / particles as f32)
        }
    };
}

/// Metal-accelerated Brownian motion estimators.
pub mod bm;
/// Metal-accelerated Lévy process estimators.
pub mod levy;
/// Metal-accelerated Ornstein-Uhlenbeck process estimators.
pub mod ou;
/// Metal-accelerated random number generators.
pub mod random;
