use crate::XResult;
use cudarc::driver::{CudaContext, CudaStream, LaunchConfig};
use std::sync::{Arc, LazyLock};

pub(crate) static CUDA_CTX: LazyLock<XResult<Arc<CudaContext>>> =
    LazyLock::new(|| Ok(CudaContext::new(0)?));

pub(crate) static CUDA_STREAM: LazyLock<XResult<Arc<CudaStream>>> =
    LazyLock::new(|| Ok(CUDA_CTX.as_ref()?.default_stream()));

#[inline]
pub(crate) fn config(particles: usize) -> LaunchConfig {
    let block_size = 256;
    let grid_size = particles.div_ceil(block_size);
    LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub trait GPUMoment {
    /// Get the raw moment of the simulation (GPU version)
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    fn raw_moment_gpu(
        &self,
        duration: f32,
        order: i32,
        particles: usize,
        time_step: f32,
    ) -> XResult<f32>;

    /// Get the central moment of the simulation (GPU version)
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    fn central_moment_gpu(
        &self,
        duration: f32,
        order: i32,
        particles: usize,
        time_step: f32,
    ) -> XResult<f32>;

    /// Get the mean of the simulation (GPU version)
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    fn mean_gpu(&self, duration: f32, particles: usize, time_step: f32) -> XResult<f32>;

    /// Get the standard deviation of the simulation (GPU version)
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    fn msd_gpu(&self, duration: f32, particles: usize, time_step: f32) -> XResult<f32>;

    /// Get the fractional moment of the simulation (GPU version)
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    /// * `duration` - The duration of the simulation.
    fn frac_raw_moment_gpu(
        &self,
        duration: f32,
        order: f32,
        particles: usize,
        time_step: f32,
    ) -> XResult<f32>;

    /// Get the fractional central moment of the simulation (GPU version)
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the simulation.
    /// * `duration` - The duration of the simulation.
    fn frac_central_moment_gpu(
        &self,
        duration: f32,
        order: f32,
        particles: usize,
        time_step: f32,
    ) -> XResult<f32>;
}

#[macro_export]
/// Macro to generate GPU-accelerated moment calculation functions
macro_rules! subscribe_gpu_function {
    ($module:expr, $func_name:ident, $kernel_name:expr, ($( $param_name:ident: $param_type:ty ),+ $(,)?)) => {
        #[allow(clippy::too_many_arguments)]
        fn $func_name(
            $(
                $param_name: $param_type,
            )+
            particles: usize,
        ) -> XResult<f32> {
            let stream = $crate::gpu::CUDA_STREAM.as_ref()?;
            let kernel = $kernel_name.as_ref()?;
            let mut device_out = stream.alloc_zeros::<f32>(1)?;
            let cfg = $crate::gpu::config(particles);

            let mut rng = rand::rng();
            use rand::Rng;
            let seed: u64 = rng.random();

            let mut builder = stream.launch_builder(kernel);
            use cudarc::driver::PushKernelArg;
            builder.arg(&mut device_out);

            $(
                builder.arg(&$param_name);
            )+

            builder.arg(&particles);
            builder.arg(&seed);

            unsafe {
                builder.launch(cfg)?;
            }

            let out_host = stream.clone_dtoh(&device_out)?;
            let sum = out_host[0];
            Ok(sum / particles as f32)
        }
    };
}

#[macro_export]
/// Macro to generate GPU-accelerated central moment calculation functions
macro_rules! subscribe_central_moment_gpu_function {
    ($module:expr, $func_name:ident, $kernel_name:expr, ($( $param_name:ident: $param_type:ty ),+ $(,)?), $order_type:ty) => {
        #[allow(clippy::too_many_arguments)]
        fn $func_name(
            $(
                $param_name: $param_type,
            )+
            order: $order_type,
            particles: usize,
        ) -> XResult<f32> {
            let stream = $crate::gpu::CUDA_STREAM.as_ref()?;
            let kernel = $kernel_name.as_ref()?;
            let mut device_out = stream.alloc_zeros::<f32>(1)?;
            let cfg = $crate::gpu::config(particles);

            let mut rng = rand::rng();
            use rand::Rng;
            let seed: u64 = rng.random();

            let mean = mean($($param_name,)+ particles)?;

            let mut builder = stream.launch_builder(kernel);
            use cudarc::driver::PushKernelArg;
            builder.arg(&mut device_out);
            builder.arg(&mean);
            builder.arg(&order);

            $(
                builder.arg(&$param_name);
            )+

            builder.arg(&particles);
            builder.arg(&seed);

            unsafe {
                builder.launch(cfg)?;
            }

            let out_host = stream.clone_dtoh(&device_out)?;
            let sum = out_host[0];
            Ok(sum / particles as f32)
        }
    };
}

pub mod bm;
pub mod levy;
pub mod ou;
pub mod stable;
