use super::CUDA_CTX;
use crate::{XResult, gpu::GPUMoment, simulation::continuous::Bm};
use cuda_kernel::BM_PTX;
use cudarc::{
    driver::{CudaModule, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use std::sync::{Arc, LazyLock};

static MODULE: LazyLock<XResult<Arc<CudaModule>>> = LazyLock::new(|| {
    let ctx = CUDA_CTX.as_ref()?;
    let module = ctx.load_module(Ptx::from(BM_PTX))?;
    Ok(module)
});

pub fn bm_mean(
    start_position: f32,
    diffusivity: f32,
    duration: f32,
    time_step: f32,
    particles: usize,
) -> XResult<f32> {
    let ctx = CUDA_CTX.as_ref()?;
    let stream = ctx.default_stream();
    let module = MODULE.as_ref()?;
    let mean = module.load_function("bm_mean")?;

    let mut device_out = stream.alloc_zeros::<f32>(1)?;

    let block_size = 256;
    let grid_size = particles.div_ceil(block_size);
    let cfg = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let seed = std::time::SystemTime::now().elapsed()?.as_secs();

    let mut builder = stream.launch_builder(&mean);
    builder.arg(&mut device_out);
    builder.arg(&start_position);
    builder.arg(&diffusivity);
    builder.arg(&duration);
    builder.arg(&time_step);
    builder.arg(&particles);
    builder.arg(&seed);

    unsafe {
        builder.launch(cfg)?;
    }

    let out_host = stream.clone_dtoh(&device_out)?;
    let sum = out_host[0];
    Ok(sum / particles as f32)
}

pub fn bm_msd(diffusivity: f32, duration: f32, time_step: f32, particles: usize) -> XResult<f32> {
    let ctx = CUDA_CTX.as_ref()?;
    let stream = ctx.default_stream();
    let module = MODULE.as_ref()?;
    let msd = module.load_function("bm_msd")?;

    let mut device_out = stream.alloc_zeros::<f32>(1)?;

    let block_size = 256;
    let grid_size = particles.div_ceil(block_size);
    let cfg = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let seed = std::time::SystemTime::now().elapsed()?.as_secs();

    let mut builder = stream.launch_builder(&msd);
    builder.arg(&mut device_out);
    builder.arg(&diffusivity);
    builder.arg(&duration);
    builder.arg(&time_step);
    builder.arg(&particles);
    builder.arg(&seed);

    unsafe {
        builder.launch(cfg)?;
    }

    let out_host = stream.clone_dtoh(&device_out)?;
    let sum = out_host[0];
    Ok(sum / particles as f32)
}

pub fn bm_moment(
    start_position: f32,
    diffusivity: f32,
    order: i32,
    central: bool,
    duration: f32,
    time_step: f32,
    particles: usize,
) -> XResult<f32> {
    let mean = bm_mean(start_position, diffusivity, duration, time_step, particles)?;
    let ctx = CUDA_CTX.as_ref()?;
    let stream = ctx.default_stream();
    let module = MODULE.as_ref()?;

    let mut device_out = stream.alloc_zeros::<f32>(1)?;

    let block_size = 256;
    let grid_size = particles.div_ceil(block_size);
    let cfg = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let moment_func = if central {
        module.load_function("bm_central_moment")?
    } else {
        module.load_function("bm_raw_moment")?
    };

    let mut builder = stream.launch_builder(&moment_func);

    let seed = std::time::SystemTime::now().elapsed()?.as_secs();

    builder.arg(&mut device_out);
    if central {
        builder.arg(&mean);
    }
    builder.arg(&start_position);
    builder.arg(&diffusivity);
    builder.arg(&order);
    builder.arg(&duration);
    builder.arg(&time_step);
    builder.arg(&particles);
    builder.arg(&seed);

    unsafe {
        builder.launch(cfg)?;
    }

    let out_host = stream.clone_dtoh(&device_out)?;
    let sum = out_host[0];
    Ok(sum / particles as f32)
}

impl GPUMoment for Bm {
    fn central_moment_gpu(
        &self,
        duration: f32,
        order: i32,
        particles: usize,
        time_step: f32,
    ) -> XResult<f32> {
        bm_moment(
            self.get_start_position() as f32,
            self.get_diffusion_coefficient() as f32,
            order,
            true,
            duration,
            time_step,
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
        bm_moment(
            self.get_start_position() as f32,
            self.get_diffusion_coefficient() as f32,
            order,
            false,
            duration,
            time_step,
            particles,
        )
    }

    fn mean_gpu(&self, duration: f32, particles: usize, time_step: f32) -> XResult<f32> {
        bm_mean(
            self.get_start_position() as f32,
            self.get_diffusion_coefficient() as f32,
            duration,
            time_step,
            particles,
        )
    }

    fn msd_gpu(&self, duration: f32, particles: usize, time_step: f32) -> XResult<f32> {
        bm_msd(
            self.get_diffusion_coefficient() as f32,
            duration,
            time_step,
            particles,
        )
    }
}
