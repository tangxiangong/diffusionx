use super::{CUDA_CTX, config, load_kernel};
use crate::{XResult, gpu::GPUMoment, simulation::continuous::Bm};
use cuda_kernel::BM_PTX;
use cudarc::{
    driver::{CudaModule, PushKernelArg},
    nvrtc::Ptx,
};
use std::sync::{Arc, LazyLock};

static MODULE: LazyLock<XResult<Arc<CudaModule>>> = LazyLock::new(|| {
    let ctx = CUDA_CTX.as_ref()?;
    let module = ctx.load_module(Ptx::from(BM_PTX))?;
    Ok(module)
});

fn mean(
    start_position: f32,
    diffusivity: f32,
    duration: f32,
    time_step: f32,
    particles: usize,
) -> XResult<f32> {
    let (stream, kernel) = load_kernel(&MODULE, "bm_mean")?;
    let mut device_out = stream.alloc_zeros::<f32>(1)?;
    let cfg = config(particles);

    let seed = std::time::SystemTime::now().elapsed()?.as_secs();

    let mut builder = stream.launch_builder(&kernel);
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

fn msd(diffusivity: f32, duration: f32, time_step: f32, particles: usize) -> XResult<f32> {
    let (stream, kernel) = load_kernel(&MODULE, "bm_msd")?;
    let mut device_out = stream.alloc_zeros::<f32>(1)?;
    let cfg = config(particles);

    let seed = std::time::SystemTime::now().elapsed()?.as_secs();

    let mut builder = stream.launch_builder(&kernel);
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

fn raw_moment(
    start_position: f32,
    diffusivity: f32,
    order: i32,
    duration: f32,
    time_step: f32,
    particles: usize,
) -> XResult<f32> {
    let (stream, kernel) = load_kernel(&MODULE, "bm_raw_moment")?;
    let mut device_out = stream.alloc_zeros::<f32>(1)?;
    let cfg = config(particles);

    let seed = std::time::SystemTime::now().elapsed()?.as_secs();

    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut device_out);
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

fn central_moment(
    start_position: f32,
    diffusivity: f32,
    order: i32,
    duration: f32,
    time_step: f32,
    particles: usize,
) -> XResult<f32> {
    let (stream, kernel) = load_kernel(&MODULE, "bm_central_moment")?;
    let mut device_out = stream.alloc_zeros::<f32>(1)?;
    let cfg = config(particles);

    let seed = std::time::SystemTime::now().elapsed()?.as_secs();
    let mean = mean(start_position, diffusivity, duration, time_step, particles)?;

    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut device_out);
    builder.arg(&mean);
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

fn frac_raw_moment(
    start_position: f32,
    diffusivity: f32,
    order: f32,
    duration: f32,
    time_step: f32,
    particles: usize,
) -> XResult<f32> {
    let (stream, kernel) = load_kernel(&MODULE, "bm_raw_moment")?;
    let mut device_out = stream.alloc_zeros::<f32>(1)?;
    let cfg = config(particles);

    let seed = std::time::SystemTime::now().elapsed()?.as_secs();

    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut device_out);
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

fn frac_central_moment(
    start_position: f32,
    diffusivity: f32,
    order: i32,
    duration: f32,
    time_step: f32,
    particles: usize,
) -> XResult<f32> {
    let (stream, kernel) = load_kernel(&MODULE, "bm_central_moment")?;
    let mut device_out = stream.alloc_zeros::<f32>(1)?;
    let cfg = config(particles);

    let seed = std::time::SystemTime::now().elapsed()?.as_secs();
    let mean = mean(start_position, diffusivity, duration, time_step, particles)?;

    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut device_out);
    builder.arg(&mean);
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
        central_moment(
            self.get_start_position() as f32,
            self.get_diffusion_coefficient() as f32,
            order,
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
        raw_moment(
            self.get_start_position() as f32,
            self.get_diffusion_coefficient() as f32,
            order,
            duration,
            time_step,
            particles,
        )
    }

    fn mean_gpu(&self, duration: f32, particles: usize, time_step: f32) -> XResult<f32> {
        mean(
            self.get_start_position() as f32,
            self.get_diffusion_coefficient() as f32,
            duration,
            time_step,
            particles,
        )
    }

    fn msd_gpu(&self, duration: f32, particles: usize, time_step: f32) -> XResult<f32> {
        msd(
            self.get_diffusion_coefficient() as f32,
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
            self.get_start_position() as f32,
            self.get_diffusion_coefficient() as f32,
            order as i32,
            duration,
            time_step,
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
            self.get_start_position() as f32,
            self.get_diffusion_coefficient() as f32,
            order,
            duration,
            time_step,
            particles,
        )
    }
}
