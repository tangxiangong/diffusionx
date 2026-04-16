use crate::{
    FloatExt, XError, XResult,
    gpu::{CUDA_CTX, GPUMoment, LEVY_PTX},
    simulation::continuous::{Bm, Levy},
};
use cudarc::{
    driver::{CudaFunction, CudaModule},
    nvrtc::Ptx,
};
use std::sync::{Arc, LazyLock};

static MODULE: LazyLock<XResult<Arc<CudaModule>>> = LazyLock::new(|| {
    let ctx = CUDA_CTX.as_ref()?;
    let module = ctx.load_module(Ptx::from(LEVY_PTX))?;
    Ok(module)
});

static MEAN_KERNEL: LazyLock<XResult<CudaFunction>> = LazyLock::new(|| {
    let module = MODULE.as_ref()?;
    let kernel = module.load_function("mean")?;
    Ok(kernel)
});

static RAW_MOMENT_KERNEL: LazyLock<XResult<CudaFunction>> = LazyLock::new(|| {
    let module = MODULE.as_ref()?;
    let kernel = module.load_function("raw_moment")?;
    Ok(kernel)
});

static FRAC_RAW_MOMENT_KERNEL: LazyLock<XResult<CudaFunction>> = LazyLock::new(|| {
    let module = MODULE.as_ref()?;
    let kernel = module.load_function("frac_raw_moment")?;
    Ok(kernel)
});

static FRAC_CENTRAL_MOMENT_KERNEL: LazyLock<XResult<CudaFunction>> = LazyLock::new(|| {
    let module = MODULE.as_ref()?;
    let kernel = module.load_function("frac_central_moment")?;
    Ok(kernel)
});

subscribe_gpu_function!(MODULE, mean, MEAN_KERNEL, (alpha: f32, start_position: f32, duration: f32, time_step: f32, inv_alpha: f32, one_minus_alpha_div_alpha: f32));

subscribe_gpu_function!(MODULE, raw_moment, RAW_MOMENT_KERNEL, (alpha: f32, start_position: f32, order: i32, duration: f32, time_step: f32, inv_alpha: f32, one_minus_alpha_div_alpha: f32));

subscribe_gpu_function!(MODULE, frac_raw_moment, FRAC_RAW_MOMENT_KERNEL, (alpha: f32, start_position: f32, order: f32, duration: f32, time_step: f32, inv_alpha: f32, one_minus_alpha_div_alpha: f32));

subscribe_central_moment_gpu_function!(MODULE, frac_central_moment, FRAC_CENTRAL_MOMENT_KERNEL, (alpha: f32, start_position: f32, duration: f32, time_step: f32, inv_alpha: f32, one_minus_alpha_div_alpha: f32), f32);

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
