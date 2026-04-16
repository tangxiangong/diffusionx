use crate::{
    FloatExt, XResult,
    gpu::{
        GPUMoment,
        metal::{OU_METALLIB, load_library},
    },
    simulation::continuous::OrnsteinUhlenbeck as OU,
};
use std::sync::LazyLock;

static LIBRARY: LazyLock<XResult<metal::Library>> = LazyLock::new(|| load_library(OU_METALLIB));

subscribe_metal_gpu_function!(LIBRARY, mean, "mean", (start_position: f32, theta: f32, sigma: f32, duration: f32, time_step: f32));

subscribe_metal_gpu_function!(LIBRARY, msd, "msd", (start_position: f32, theta: f32, sigma: f32, duration: f32, time_step: f32));

subscribe_metal_gpu_function!(LIBRARY, raw_moment, "raw_moment", (start_position: f32, theta: f32, sigma: f32, order: i32, duration: f32, time_step: f32));

subscribe_metal_gpu_function!(LIBRARY, frac_raw_moment, "frac_raw_moment", (start_position: f32, theta: f32, sigma: f32, order: f32, duration: f32, time_step: f32));

subscribe_metal_central_moment_gpu_function!(LIBRARY, central_moment, "central_moment", (start_position: f32, theta: f32, sigma: f32, duration: f32, time_step: f32), i32);

subscribe_metal_central_moment_gpu_function!(LIBRARY, frac_central_moment, "frac_central_moment", (start_position: f32, theta: f32, sigma: f32, duration: f32, time_step: f32), f32);

impl<T: FloatExt> GPUMoment for OU<T> {
    fn central_moment_gpu(
        &self,
        duration: f32,
        order: i32,
        particles: usize,
        time_step: f32,
    ) -> XResult<f32> {
        central_moment(
            self.get_start_position().to_f32().unwrap(),
            self.get_theta().to_f32().unwrap(),
            self.get_sigma().to_f32().unwrap(),
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
            self.get_theta().to_f32().unwrap(),
            self.get_sigma().to_f32().unwrap(),
            order,
            duration,
            time_step,
            particles,
        )
    }

    fn mean_gpu(&self, duration: f32, particles: usize, time_step: f32) -> XResult<f32> {
        mean(
            self.get_start_position().to_f32().unwrap(),
            self.get_theta().to_f32().unwrap(),
            self.get_sigma().to_f32().unwrap(),
            duration,
            time_step,
            particles,
        )
    }

    fn msd_gpu(&self, duration: f32, particles: usize, time_step: f32) -> XResult<f32> {
        msd(
            self.get_start_position().to_f32().unwrap(),
            self.get_theta().to_f32().unwrap(),
            self.get_sigma().to_f32().unwrap(),
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
            self.get_theta().to_f32().unwrap(),
            self.get_sigma().to_f32().unwrap(),
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
            self.get_theta().to_f32().unwrap(),
            self.get_sigma().to_f32().unwrap(),
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
    use crate::simulation::continuous::OrnsteinUhlenbeck;

    #[test]
    fn test_gpu_moment() {
        let ou = OrnsteinUhlenbeck::<f32>::default();
        ou.mean_gpu(1.0, 100, 0.1).unwrap();
        ou.msd_gpu(1.0, 100, 0.1).unwrap();
        ou.raw_moment_gpu(1.0, 2, 100, 0.1).unwrap();
        ou.frac_raw_moment_gpu(1.0, 1.4, 100, 0.1).unwrap();
        ou.central_moment_gpu(1.0, 2, 100, 0.1).unwrap();
        ou.frac_central_moment_gpu(1.0, 1.5, 100, 0.1).unwrap();
    }
}
