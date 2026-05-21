use crate::{
    FloatExt, XError, XResult,
    gpu::{
        GPUMoment,
        metal::{LEVY_METALLIB, MetalLibrary, load_library},
    },
    simulation::continuous::{Bm, Levy},
};
use objc2::rc::Retained;
use std::sync::LazyLock;

static LIBRARY: LazyLock<XResult<Retained<MetalLibrary>>> =
    LazyLock::new(|| load_library(LEVY_METALLIB));

subscribe_metal_gpu_function!(LIBRARY, mean, "mean", (alpha: f32, start_position: f32, duration: f32, time_step: f32, inv_alpha: f32, one_minus_alpha_div_alpha: f32));

subscribe_metal_gpu_function!(LIBRARY, raw_moment, "raw_moment", (alpha: f32, start_position: f32, order: i32, duration: f32, time_step: f32, inv_alpha: f32, one_minus_alpha_div_alpha: f32));

subscribe_metal_gpu_function!(LIBRARY, frac_raw_moment, "frac_raw_moment", (alpha: f32, start_position: f32, order: f32, duration: f32, time_step: f32, inv_alpha: f32, one_minus_alpha_div_alpha: f32));

subscribe_metal_frac_central_moment_gpu_function!(
    LIBRARY,
    frac_central_moment,
    "frac_central_moment",
    (
        alpha: f32,
        start_position: f32,
        duration: f32,
        time_step: f32,
        inv_alpha: f32,
        one_minus_alpha_div_alpha: f32
    ),
    before_order = (alpha, start_position),
    after_order = (
        duration,
        time_step,
        inv_alpha,
        one_minus_alpha_div_alpha
    )
);

impl<T: FloatExt> GPUMoment for Levy<T> {
    fn central_moment_gpu(
        &self,
        duration: f32,
        order: i32,
        particles: usize,
        time_step: f32,
    ) -> XResult<f32> {
        let alpha = self.get_alpha().to_f32().unwrap();
        if (alpha - 2.0).abs() <= f32::EPSILON {
            let bm = Bm::<f32>::default();
            return bm.central_moment_gpu(duration, order, particles, time_step);
        }
        if alpha <= order as f32 || order < 0 {
            return Err(XError::Other("Not Exist".into()));
        }
        if order == 1 { Ok(0.0) } else { Ok(1.0) }
    }

    fn raw_moment_gpu(
        &self,
        duration: f32,
        order: i32,
        particles: usize,
        time_step: f32,
    ) -> XResult<f32> {
        let alpha = self.get_alpha().to_f32().unwrap();
        if (alpha - 2.0).abs() < f32::EPSILON {
            let bm = Bm::<f32>::default();
            return bm.raw_moment_gpu(duration, order, particles, time_step);
        }

        if alpha <= order as f32 {
            return Err(XError::Other("Not Exist".into()));
        }

        let inv_alpha = 1.0 / alpha;
        let one_minus_alpha_div_alpha = (1.0 - alpha) / alpha;

        if order == 1 {
            mean(
                alpha,
                self.get_start_position().to_f32().unwrap(),
                duration,
                time_step,
                inv_alpha,
                one_minus_alpha_div_alpha,
                particles,
            )
        } else if order == 0 {
            Ok(1.0)
        } else {
            raw_moment(
                alpha,
                self.get_start_position().to_f32().unwrap(),
                order,
                duration,
                time_step,
                inv_alpha,
                one_minus_alpha_div_alpha,
                particles,
            )
        }
    }

    fn msd_gpu(&self, duration: f32, particles: usize, time_step: f32) -> XResult<f32> {
        let alpha = self.get_alpha().to_f32().unwrap();
        if (alpha - 2.0).abs() < f32::EPSILON {
            let bm = Bm::<f32>::default();
            bm.msd_gpu(duration, particles, time_step)
        } else {
            Err(XError::Other("Not Exist".into()))
        }
    }

    fn mean_gpu(&self, duration: f32, particles: usize, time_step: f32) -> XResult<f32> {
        let alpha = self.get_alpha().to_f32().unwrap();
        if (alpha - 2.0).abs() < f32::EPSILON {
            let bm = Bm::<f32>::default();
            return bm.mean_gpu(duration, particles, time_step);
        }

        if alpha < 1.0 {
            return Err(XError::Other("Not Exist".into()));
        }

        let inv_alpha = 1.0 / alpha;
        let one_minus_alpha_div_alpha = (1.0 - alpha) / alpha;
        mean(
            alpha,
            self.get_start_position().to_f32().unwrap(),
            duration,
            time_step,
            inv_alpha,
            one_minus_alpha_div_alpha,
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
        let alpha = self.get_alpha().to_f32().unwrap();
        if (alpha - 2.0).abs() < f32::EPSILON {
            let bm = Bm::<f32>::default();
            return bm.frac_central_moment_gpu(duration, order, particles, time_step);
        }
        if alpha < 1.0 {
            return Err(XError::Other("Not Exist".into()));
        }
        let inv_alpha = 1.0 / alpha;
        let one_minus_alpha_div_alpha = (1.0 - alpha) / alpha;
        frac_central_moment(
            alpha,
            self.get_start_position().to_f32().unwrap(),
            duration,
            time_step,
            inv_alpha,
            one_minus_alpha_div_alpha,
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
        let alpha = self.get_alpha().to_f32().unwrap();
        if (alpha - 2.0).abs() < f32::EPSILON {
            let bm = Bm::<f32>::default();
            return bm.frac_raw_moment_gpu(duration, order, particles, time_step);
        }
        if alpha < order {
            return Err(XError::Other("Not Exist".into()));
        }
        let inv_alpha = 1.0 / alpha;
        let one_minus_alpha_div_alpha = (1.0 - alpha) / alpha;
        frac_raw_moment(
            alpha,
            self.get_start_position().to_f32().unwrap(),
            order,
            duration,
            time_step,
            inv_alpha,
            one_minus_alpha_div_alpha,
            particles,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::GPUMoment;
    use crate::simulation::continuous::Levy;

    #[test]
    fn test_gpu_moment() {
        // A non-Gaussian alpha so the Levy metallib kernels are dispatched
        // instead of routing through the Brownian-motion fallback.
        let levy = Levy::<f32>::new(0.0, 1.5).unwrap();

        let mean = levy.mean_gpu(1.0, 100, 0.1).unwrap();
        assert!(mean.is_finite());

        let frac_raw = levy.frac_raw_moment_gpu(1.0, 1.2, 100, 0.1).unwrap();
        assert!(frac_raw.is_finite());

        let frac_central = levy.frac_central_moment_gpu(1.0, 1.2, 100, 0.1).unwrap();
        assert!(frac_central.is_finite());
    }
}
