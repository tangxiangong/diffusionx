use crate::XResult;
use cudarc::driver::CudaContext;
use std::sync::{Arc, LazyLock};

static CUDA_CTX: LazyLock<XResult<Arc<CudaContext>>> = LazyLock::new(|| Ok(CudaContext::new(0)?));

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
