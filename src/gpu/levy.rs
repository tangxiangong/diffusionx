use super::{CUDA_CTX, config, load_kernel};
use crate::{
    XError, XResult,
    gpu::GPUMoment,
    simulation::continuous::{Bm, Levy},
    subscribe_gpu_function,
};
use cuda_kernel::LEVY_PTX;
use cudarc::{
    driver::{CudaModule, PushKernelArg},
    nvrtc::Ptx,
};
use std::sync::{Arc, LazyLock};

static MODULE: LazyLock<XResult<Arc<CudaModule>>> = LazyLock::new(|| {
    let ctx = CUDA_CTX.as_ref()?;
    let module = ctx.load_module(Ptx::from(LEVY_PTX))?;
    Ok(module)
});

subscribe_gpu_function!(MODULE, mean, "mean", (alpha: f32, start_position: f32, duration: f32, time_step: f32, inv_alpha: f32, one_minus_alpha_div_alpha: f32));

subscribe_gpu_function!(MODULE, raw_moment, "raw_moment", (alpha: f32, start_position: f32, order: i32, duration: f32, time_step: f32, inv_alpha: f32, one_minus_alpha_div_alpha: f32));

subscribe_gpu_function!(MODULE, frac_raw_moment, "frac_raw_moment", (alpha: f32, start_position: f32, order: f32, duration: f32, time_step: f32, inv_alpha: f32, one_minus_alpha_div_alpha: f32));

subscribe_central_moment_gpu_function!(MODULE, frac_central_moment, "frac_central_moment", (alpha: f32, start_position: f32, duration: f32, time_step: f32, inv_alpha: f32, one_minus_alpha_div_alpha: f32), f32);

impl GPUMoment for Levy {
    fn central_moment_gpu(
        &self,
        duration: f32,
        order: i32,
        particles: usize,
        time_step: f32,
    ) -> XResult<f32> {
        let alpha = self.get_alpha() as f32;
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
        let alpha = self.get_alpha() as f32;
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
                self.get_start_position() as f32,
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
                self.get_start_position() as f32,
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
        let alpha = self.get_alpha() as f32;
        if (alpha - 2.0).abs() < f32::EPSILON {
            let bm = Bm::<f32>::default();
            bm.msd_gpu(duration, particles, time_step)
        } else {
            Err(XError::Other("Not Exist".into()))
        }
    }

    fn mean_gpu(&self, duration: f32, particles: usize, time_step: f32) -> XResult<f32> {
        let alpha = self.get_alpha() as f32;
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
            self.get_start_position() as f32,
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
        let alpha = self.get_alpha() as f32;
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
            self.get_start_position() as f32,
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
        let alpha = self.get_alpha() as f32;
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
            self.get_start_position() as f32,
            order,
            duration,
            time_step,
            inv_alpha,
            one_minus_alpha_div_alpha,
            particles,
        )
    }
}
