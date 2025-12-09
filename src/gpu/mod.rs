use crate::XResult;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(feature = "metal")]
pub mod metal;
#[cfg(feature = "metal")]
pub use metal::*;

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
