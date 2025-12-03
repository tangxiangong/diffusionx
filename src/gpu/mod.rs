use crate::XResult;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig};
use std::sync::{Arc, LazyLock};

static CUDA_CTX: LazyLock<XResult<Arc<CudaContext>>> = LazyLock::new(|| Ok(CudaContext::new(0)?));

pub(crate) fn load_kernel(
    cuda_module: &LazyLock<XResult<Arc<CudaModule>>>,
    kernel_name: &str,
) -> XResult<(Arc<CudaStream>, CudaFunction)> {
    let ctx = CUDA_CTX.as_ref()?;
    let stream = ctx.default_stream();
    let module = cuda_module.as_ref()?;
    let kernel = module.load_function(kernel_name)?;
    Ok((stream, kernel))
}

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

    fn mean_gpu(&self, duration: f32, particles: usize, time_step: f32) -> XResult<f32>;
    fn msd_gpu(&self, duration: f32, particles: usize, time_step: f32) -> XResult<f32>;
}

pub mod bm;
pub mod stable;
