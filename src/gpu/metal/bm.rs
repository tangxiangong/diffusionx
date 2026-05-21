use crate::{
    FloatExt, XResult,
    gpu::{
        GPUMoment,
        metal::{BM_METALLIB, MetalLibrary, load_library},
    },
    simulation::continuous::Bm,
};
use objc2::rc::Retained;
use std::sync::LazyLock;

static LIBRARY: LazyLock<XResult<Retained<MetalLibrary>>> =
    LazyLock::new(|| load_library(BM_METALLIB));

subscribe_metal_gpu_function!(LIBRARY, mean, "mean", (start_position: f32, diffusivity: f32, duration: f32, time_step: f32));

subscribe_metal_gpu_function!(LIBRARY, msd, "msd", (start_position: f32, diffusivity: f32, duration: f32, time_step: f32));

subscribe_metal_gpu_function!(LIBRARY, raw_moment, "raw_moment", (start_position: f32, diffusivity: f32, order: i32, duration: f32, time_step: f32));

subscribe_metal_gpu_function!(LIBRARY, frac_raw_moment, "frac_raw_moment", (start_position: f32, diffusivity: f32, order: f32, duration: f32, time_step: f32));

subscribe_metal_central_moment_gpu_function!(LIBRARY, central_moment, "central_moment", (start_position: f32, diffusivity: f32, duration: f32, time_step: f32), i32);

subscribe_metal_frac_central_moment_gpu_function!(
    LIBRARY,
    frac_central_moment,
    "frac_central_moment",
    (
        start_position: f32,
        diffusivity: f32,
        duration: f32,
        time_step: f32
    ),
    before_order = (),
    after_order = (start_position, diffusivity, duration, time_step)
);

impl<T: FloatExt> GPUMoment for Bm<T> {
    fn central_moment_gpu(
        &self,
        duration: f32,
        order: i32,
        particles: usize,
        time_step: f32,
    ) -> XResult<f32> {
        central_moment(
            self.get_start_position().to_f32().unwrap(),
            self.get_diffusion_coefficient().to_f32().unwrap(),
            duration,
            time_step,
            order,
            particles,
        )
    }

    fn raw_moment_gpu(
        &self,
        duration: f32,
        order: i32,
        particles: usize,
        time_step: f32,
    ) -> XResult<f32> {
        raw_moment(
            self.get_start_position().to_f32().unwrap(),
            self.get_diffusion_coefficient().to_f32().unwrap(),
            order,
            duration,
            time_step,
            particles,
        )
    }

    fn mean_gpu(&self, duration: f32, particles: usize, time_step: f32) -> XResult<f32> {
        mean(
            self.get_start_position().to_f32().unwrap(),
            self.get_diffusion_coefficient().to_f32().unwrap(),
            duration,
            time_step,
            particles,
        )
    }

    fn msd_gpu(&self, duration: f32, particles: usize, time_step: f32) -> XResult<f32> {
        msd(
            self.get_start_position().to_f32().unwrap(),
            self.get_diffusion_coefficient().to_f32().unwrap(),
            duration,
            time_step,
            particles,
        )
    }

    fn frac_central_moment_gpu(
        &self,
        duration: f32,
        order: f32,
        particles: usize,
        time_step: f32,
    ) -> XResult<f32> {
        frac_central_moment(
            self.get_start_position().to_f32().unwrap(),
            self.get_diffusion_coefficient().to_f32().unwrap(),
            duration,
            time_step,
            order,
            particles,
        )
    }

    fn frac_raw_moment_gpu(
        &self,
        duration: f32,
        order: f32,
        particles: usize,
        time_step: f32,
    ) -> XResult<f32> {
        frac_raw_moment(
            self.get_start_position().to_f32().unwrap(),
            self.get_diffusion_coefficient().to_f32().unwrap(),
            order,
            duration,
            time_step,
            particles,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::GPUMoment;
    use crate::simulation::continuous::Bm;

    #[test]
    fn test_gpu_moment() {
        let bm = Bm::<f32>::default();
        bm.mean_gpu(1.0, 100, 0.1).unwrap();
        bm.msd_gpu(1.0, 100, 0.1).unwrap();
        bm.raw_moment_gpu(1.0, 2, 100, 0.1).unwrap();
        bm.frac_raw_moment_gpu(1.0, 1.4, 100, 0.1).unwrap();
        bm.central_moment_gpu(1.0, 2, 100, 0.1).unwrap();
        bm.frac_central_moment_gpu(1.0, 1.5, 100, 0.1).unwrap();
    }
}
