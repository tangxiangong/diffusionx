use crate::{
    FloatExt, XResult,
    gpu::{BM_PTX, CUDA_CTX, GPUMoment},
    simulation::continuous::Bm,
};
use cudarc::{
    driver::{CudaFunction, CudaModule},
    nvrtc::Ptx,
};
use std::sync::{Arc, LazyLock};

static MODULE: LazyLock<XResult<Arc<CudaModule>>> = LazyLock::new(|| {
    let ctx = CUDA_CTX.as_ref()?;
    let module = ctx.load_module(Ptx::from(BM_PTX))?;
    Ok(module)
});

static MEAN_KERNEL: LazyLock<XResult<CudaFunction>> = LazyLock::new(|| {
    let module = MODULE.as_ref()?;
    let kernel = module.load_function("mean")?;
    Ok(kernel)
});

static MSD_KERNEL: LazyLock<XResult<CudaFunction>> = LazyLock::new(|| {
    let module = MODULE.as_ref()?;
    let kernel = module.load_function("msd")?;
    Ok(kernel)
});

static RAW_MOMENT_KERNEL: LazyLock<XResult<CudaFunction>> = LazyLock::new(|| {
    let module = MODULE.as_ref()?;
    let kernel = module.load_function("raw_moment")?;
    Ok(kernel)
});

static FRAC_RAW_KERNEL: LazyLock<XResult<CudaFunction>> = LazyLock::new(|| {
    let module = MODULE.as_ref()?;
    let kernel = module.load_function("frac_raw_moment")?;
    Ok(kernel)
});

static CENTRAL_MOMENT_KERNEL: LazyLock<XResult<CudaFunction>> = LazyLock::new(|| {
    let module = MODULE.as_ref()?;
    let kernel = module.load_function("central_moment")?;
    Ok(kernel)
});

static FRAC_CENTRAL_KERNEL: LazyLock<XResult<CudaFunction>> = LazyLock::new(|| {
    let module = MODULE.as_ref()?;
    let kernel = module.load_function("frac_central_moment")?;
    Ok(kernel)
});

subscribe_gpu_function!(MODULE, mean, MEAN_KERNEL, (start_position: f32, diffusivity: f32, duration: f32, time_step: f32));

subscribe_gpu_function!(MODULE, msd, MSD_KERNEL, (start_position: f32, diffusivity: f32, duration: f32, time_step: f32));

subscribe_gpu_function!(MODULE, raw_moment, RAW_MOMENT_KERNEL, (start_position: f32, diffusivity: f32, order: i32, duration: f32, time_step: f32));

subscribe_gpu_function!(MODULE, frac_raw_moment, FRAC_RAW_KERNEL, (start_position: f32, diffusivity: f32, order: f32, duration: f32, time_step: f32));

subscribe_central_moment_gpu_function!(MODULE, central_moment, CENTRAL_MOMENT_KERNEL, (start_position: f32, diffusivity: f32, duration: f32, time_step: f32), i32);

subscribe_central_moment_gpu_function!(MODULE, frac_central_moment, FRAC_CENTRAL_KERNEL, (start_position: f32, diffusivity: f32, duration: f32, time_step: f32), f32);

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
