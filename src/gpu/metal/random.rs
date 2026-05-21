use crate::{
    XResult,
    gpu::metal::{
        METAL_QUEUE, MetalLibrary, MetalPipeline, RANDOM_METALLIB, dispatch_threadgroups,
        end_encoding, finish_command_buffer, get_pipeline, load_library, new_command_buffer,
        new_compute_encoder, new_shared_buffer, read_buffer_vec_f32, set_buffer, set_pipeline,
        set_scalar, thread_config,
    },
};
use objc2::rc::Retained;
use rand::RngExt;
use std::sync::LazyLock;

static LIBRARY: LazyLock<XResult<Retained<MetalLibrary>>> =
    LazyLock::new(|| load_library(RANDOM_METALLIB));
static STANDARD_STABLE_PIPELINE: LazyLock<XResult<Retained<MetalPipeline>>> = LazyLock::new(|| {
    let library = LIBRARY.as_ref().map_err(Clone::clone)?;
    get_pipeline(library, "standard_stable_rand")
});
static UNIFORM_PIPELINE: LazyLock<XResult<Retained<MetalPipeline>>> = LazyLock::new(|| {
    let library = LIBRARY.as_ref().map_err(Clone::clone)?;
    get_pipeline(library, "randuniform")
});
static NORMAL_PIPELINE: LazyLock<XResult<Retained<MetalPipeline>>> = LazyLock::new(|| {
    let library = LIBRARY.as_ref().map_err(Clone::clone)?;
    get_pipeline(library, "randnormal")
});
static EXP_PIPELINE: LazyLock<XResult<Retained<MetalPipeline>>> = LazyLock::new(|| {
    let library = LIBRARY.as_ref().map_err(Clone::clone)?;
    get_pipeline(library, "randexp")
});

/// Generate standard stable random numbers on Metal GPU
pub fn standard_stable_rands(alpha: f32, beta: f32, len: usize) -> XResult<Vec<f32>> {
    let queue = METAL_QUEUE.as_ref().map_err(Clone::clone)?;
    let pipeline = STANDARD_STABLE_PIPELINE.as_ref().map_err(Clone::clone)?;

    let (inv_alpha, one_minus_alpha_div_alpha, b, s) = if (alpha - 1.0).abs() < 1e-3 {
        (0.0f32, 0.0f32, 0.0f32, 0.0f32)
    } else {
        let inv_alpha = 1.0 / alpha;
        let one_minus_alpha_div_alpha = (1.0 - alpha) * inv_alpha;
        let tmp = beta * (alpha * std::f32::consts::FRAC_PI_2).tan();
        let b = tmp.atan() * inv_alpha;
        let s = (1.0 + tmp * tmp).powf(0.5 * inv_alpha);
        (inv_alpha, one_minus_alpha_div_alpha, b, s)
    };

    let out_buffer = new_shared_buffer(len * std::mem::size_of::<f32>())?;

    let seed: u64 = rand::rng().random();
    let len_u32 = len as u32;

    let (thread_groups, threads_per_group) = thread_config(len);

    let command_buffer = new_command_buffer(queue)?;
    let encoder = new_compute_encoder(&command_buffer)?;

    set_pipeline(&encoder, pipeline);
    set_buffer(&encoder, 0, &out_buffer);
    set_scalar(&encoder, 1, &alpha);
    set_scalar(&encoder, 2, &beta);
    set_scalar(&encoder, 3, &inv_alpha);
    set_scalar(&encoder, 4, &one_minus_alpha_div_alpha);
    set_scalar(&encoder, 5, &b);
    set_scalar(&encoder, 6, &s);
    set_scalar(&encoder, 7, &len_u32);
    set_scalar(&encoder, 8, &seed);

    dispatch_threadgroups(&encoder, thread_groups, threads_per_group);
    end_encoding(&encoder);

    finish_command_buffer(&command_buffer)?;

    Ok(read_buffer_vec_f32(&out_buffer, len))
}

/// Generate uniform random numbers in (0, 1] on Metal GPU
pub fn metalrands(n: usize) -> XResult<Vec<f32>> {
    let queue = METAL_QUEUE.as_ref().map_err(Clone::clone)?;
    let pipeline = UNIFORM_PIPELINE.as_ref().map_err(Clone::clone)?;

    let out_buffer = new_shared_buffer(n * std::mem::size_of::<f32>())?;

    let seed: u64 = rand::rng().random();
    let len_u32 = n as u32;

    let (thread_groups, threads_per_group) = thread_config(n);

    let command_buffer = new_command_buffer(queue)?;
    let encoder = new_compute_encoder(&command_buffer)?;

    set_pipeline(&encoder, pipeline);
    set_buffer(&encoder, 0, &out_buffer);
    set_scalar(&encoder, 1, &len_u32);
    set_scalar(&encoder, 2, &seed);

    dispatch_threadgroups(&encoder, thread_groups, threads_per_group);
    end_encoding(&encoder);

    finish_command_buffer(&command_buffer)?;

    Ok(read_buffer_vec_f32(&out_buffer, n))
}

/// Generate normal random numbers on Metal GPU
pub fn metalrandn(n: usize, mu: f32, sigma: f32) -> XResult<Vec<f32>> {
    let queue = METAL_QUEUE.as_ref().map_err(Clone::clone)?;
    let pipeline = NORMAL_PIPELINE.as_ref().map_err(Clone::clone)?;

    let out_buffer = new_shared_buffer(n * std::mem::size_of::<f32>())?;

    let seed: u64 = rand::rng().random();
    let len_u32 = n as u32;

    let (thread_groups, threads_per_group) = thread_config(n);

    let command_buffer = new_command_buffer(queue)?;
    let encoder = new_compute_encoder(&command_buffer)?;

    set_pipeline(&encoder, pipeline);
    set_buffer(&encoder, 0, &out_buffer);
    set_scalar(&encoder, 1, &len_u32);
    set_scalar(&encoder, 2, &mu);
    set_scalar(&encoder, 3, &sigma);
    set_scalar(&encoder, 4, &seed);

    dispatch_threadgroups(&encoder, thread_groups, threads_per_group);
    end_encoding(&encoder);

    finish_command_buffer(&command_buffer)?;

    Ok(read_buffer_vec_f32(&out_buffer, n))
}

/// Generate exponential random numbers on Metal GPU
pub fn metalrandexp(n: usize) -> XResult<Vec<f32>> {
    let queue = METAL_QUEUE.as_ref().map_err(Clone::clone)?;
    let pipeline = EXP_PIPELINE.as_ref().map_err(Clone::clone)?;

    let out_buffer = new_shared_buffer(n * std::mem::size_of::<f32>())?;

    let seed: u64 = rand::rng().random();
    let len_u32 = n as u32;

    let (thread_groups, threads_per_group) = thread_config(n);

    let command_buffer = new_command_buffer(queue)?;
    let encoder = new_compute_encoder(&command_buffer)?;

    set_pipeline(&encoder, pipeline);
    set_buffer(&encoder, 0, &out_buffer);
    set_scalar(&encoder, 1, &len_u32);
    set_scalar(&encoder, 2, &seed);

    dispatch_threadgroups(&encoder, thread_groups, threads_per_group);
    end_encoding(&encoder);

    finish_command_buffer(&command_buffer)?;

    Ok(read_buffer_vec_f32(&out_buffer, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_uniform_rng_returns_finite_values_in_range() {
        let values = metalrands(128).unwrap();
        assert_eq!(values.len(), 128);
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().all(|value| *value > 0.0 && *value <= 1.0));
    }

    #[test]
    fn metal_normal_rng_returns_finite_values() {
        let values = metalrandn(128, 0.0, 1.0).unwrap();
        assert_eq!(values.len(), 128);
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn metal_exp_rng_returns_finite_non_negative_values() {
        let values = metalrandexp(128).unwrap();
        assert_eq!(values.len(), 128);
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().all(|value| *value >= 0.0));
    }

    #[test]
    fn metal_standard_stable_rng_returns_finite_values() {
        let values = standard_stable_rands(1.5, 0.0, 128).unwrap();
        assert_eq!(values.len(), 128);
        assert!(values.iter().all(|value| value.is_finite()));
    }
}
