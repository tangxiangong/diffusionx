use crate::{
    XResult,
    gpu::{CUDA_CTX, CUDA_STREAM, RANDOM_PTX, config},
};
use cudarc::{
    curand::CudaRng,
    driver::{CudaFunction, CudaModule, PushKernelArg},
    nvrtc::Ptx,
};
use rand::Rng;
use std::sync::{Arc, LazyLock};

static MODULE: LazyLock<XResult<Arc<CudaModule>>> = LazyLock::new(|| {
    let ctx = CUDA_CTX.as_ref()?;
    let module = ctx.load_module(Ptx::from(RANDOM_PTX))?;
    Ok(module)
});

static STD_STABLE_RNG: LazyLock<XResult<CudaFunction>> = LazyLock::new(|| {
    let module = MODULE.as_ref()?;
    let kernel = module.load_function("standard_stable_rand")?;
    Ok(kernel)
});

static EXP_RNG: LazyLock<XResult<CudaFunction>> = LazyLock::new(|| {
    let module = MODULE.as_ref()?;
    let kernel = module.load_function("randexp")?;
    Ok(kernel)
});

macro_rules! generate_cuda_rng {
    ($name:ident, $kernel_name:expr, $( $param_name:ident: $param_type:ty ),* $(,)?) => {
        fn $name($($param_name: $param_type,)* len: usize) -> XResult<Vec<f32>> {
            let stream = CUDA_STREAM.as_ref()?;
            let kernel = $kernel_name.as_ref()?;
            let mut device_out = stream.alloc_zeros::<f32>(len)?;
            let cfg = config(len);

            let seed: u64 = rand::rng().random();

            let mut builder = stream.launch_builder(kernel);
            builder.arg(&mut device_out);
            $(
                builder.arg(&$param_name);
            )*
            builder.arg(&len);
            builder.arg(&seed);

            unsafe {
                builder.launch(cfg)?;
            }

            let out_host = stream.clone_dtoh(&device_out)?;
            Ok(out_host)
        }
    };
}

generate_cuda_rng!(std_stable_rands_impl, STD_STABLE_RNG, alpha: f32, beta: f32, inv_alpha: f32,
    one_minus_alpha_div_alpha: f32, b: f32, s: f32);

/// Generate standard stable random numbers on the CUDA GPU.
pub fn standard_stable_rands(alpha: f32, beta: f32, len: usize) -> XResult<Vec<f32>> {
    let (inv_alpha, one_minus_alpha_div_alpha, b, s) = if (alpha - 1.0).abs() < 1e-3 {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        let inv_alpha = 1.0 / alpha;
        let one_minus_alpha_div_alpha = (1.0 - alpha) * inv_alpha;
        let tmp = beta * (alpha * std::f32::consts::FRAC_PI_2).tan();
        let b = tmp.atan() * inv_alpha;
        let s = (1.0 + tmp * tmp).powf(0.5 * inv_alpha);
        (inv_alpha, one_minus_alpha_div_alpha, b, s)
    };

    std_stable_rands_impl(alpha, beta, inv_alpha, one_minus_alpha_div_alpha, b, s, len)
}

/// Generate standard uniform random numbers on the CUDA GPU.
pub fn curands(n: usize) -> XResult<Vec<f32>> {
    let stream = CUDA_STREAM.as_ref()?;
    let rng = CudaRng::new(rand::rng().random(), stream.clone())?;
    let mut out_device = stream.alloc_zeros::<f32>(n)?;
    rng.fill_with_uniform(&mut out_device)?;
    let out_host = stream.clone_dtoh(&out_device)?;
    Ok(out_host)
}

/// Generate normal random numbers on the CUDA GPU.
pub fn curandn(n: usize, mu: f32, sigma: f32) -> XResult<Vec<f32>> {
    let stream = CUDA_STREAM.as_ref()?;
    let rng = CudaRng::new(rand::rng().random(), stream.clone())?;
    let mut out_device = stream.alloc_zeros::<f32>(n)?;
    rng.fill_with_normal(&mut out_device, mu, sigma)?;
    let out_host = stream.clone_dtoh(&out_device)?;
    Ok(out_host)
}

generate_cuda_rng!(curandexp_impl, EXP_RNG,);

/// Generate standard exponential random numbers on the CUDA GPU.
pub fn curandexp(n: usize) -> XResult<Vec<f32>> {
    curandexp_impl(n)
}
