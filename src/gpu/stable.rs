use crate::{
    XResult,
    gpu::{CUDA_CTX, CUDA_STREAM, config},
};
use cuda_kernel::STABLE_PTX;
use cudarc::{
    driver::{CudaFunction, CudaModule, PushKernelArg},
    nvrtc::Ptx,
};
use rand::Rng;
use std::sync::{Arc, LazyLock};

static MODULE: LazyLock<XResult<Arc<CudaModule>>> = LazyLock::new(|| {
    let ctx = CUDA_CTX.as_ref()?;
    let module = ctx.load_module(Ptx::from(STABLE_PTX))?;
    Ok(module)
});

static STD_STABLE_RNG: LazyLock<XResult<CudaFunction>> = LazyLock::new(|| {
    let module = MODULE.as_ref()?;
    let kernel = module.load_function("standard_stable_rand")?;
    Ok(kernel)
});

pub fn standard_stable_rands(alpha: f32, beta: f32, len: usize) -> XResult<Vec<f32>> {
    let stream = CUDA_STREAM.as_ref()?;
    let kernel = STD_STABLE_RNG.as_ref()?;
    let mut device_out = stream.alloc_zeros::<f32>(len)?;
    let cfg = config(len);

    let mut rng = rand::rng();
    let seed: u64 = rng.random();

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

    let mut builder = stream.launch_builder(kernel);
    builder.arg(&mut device_out);
    builder.arg(&alpha);
    builder.arg(&beta);
    builder.arg(&len);
    builder.arg(&seed);
    builder.arg(&inv_alpha);
    builder.arg(&one_minus_alpha_div_alpha);
    builder.arg(&b);
    builder.arg(&s);

    unsafe {
        builder.launch(cfg)?;
    }

    let out_host = stream.clone_dtoh(&device_out)?;
    Ok(out_host)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_stable_rand() {
        let r = standard_stable_rands(0.7, 0.0, 1000).unwrap();
        println!("{:?}", &r[0..10]);
    }
}
