use crate::{
    XResult,
    gpu::metal::{METAL_DEVICE, METAL_QUEUE, RANDOM_METALLIB, load_library, thread_config},
};
use metal::MTLResourceOptions;
use rand::RngExt;
use std::sync::LazyLock;

static LIBRARY: LazyLock<XResult<metal::Library>> = LazyLock::new(|| load_library(RANDOM_METALLIB));

/// Generate standard stable random numbers on Metal GPU
pub fn standard_stable_rands(alpha: f32, beta: f32, len: usize) -> XResult<Vec<f32>> {
    let device = METAL_DEVICE.as_ref()?;
    let queue = METAL_QUEUE.as_ref()?;
    let library = LIBRARY.as_ref()?;

    let function = library
        .get_function("standard_stable_rand", None)
        .map_err(|e| crate::XError::Other(format!("Function not found: {}", e)))?;

    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| crate::XError::Other(format!("Pipeline error: {}", e)))?;

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

    let out_buffer = device.new_buffer(
        (len * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let seed: u64 = rand::rng().random();
    let len_u32 = len as u32;

    let (thread_groups, threads_per_group) = thread_config(len);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&out_buffer), 0);
    encoder.set_bytes(
        1,
        std::mem::size_of::<f32>() as u64,
        &alpha as *const f32 as *const _,
    );
    encoder.set_bytes(
        2,
        std::mem::size_of::<f32>() as u64,
        &beta as *const f32 as *const _,
    );
    encoder.set_bytes(
        3,
        std::mem::size_of::<f32>() as u64,
        &inv_alpha as *const f32 as *const _,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<f32>() as u64,
        &one_minus_alpha_div_alpha as *const f32 as *const _,
    );
    encoder.set_bytes(
        5,
        std::mem::size_of::<f32>() as u64,
        &b as *const f32 as *const _,
    );
    encoder.set_bytes(
        6,
        std::mem::size_of::<f32>() as u64,
        &s as *const f32 as *const _,
    );
    encoder.set_bytes(
        7,
        std::mem::size_of::<u32>() as u64,
        &len_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        8,
        std::mem::size_of::<u64>() as u64,
        &seed as *const u64 as *const _,
    );

    encoder.dispatch_thread_groups(thread_groups, threads_per_group);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result = unsafe {
        let ptr = out_buffer.contents() as *const f32;
        std::slice::from_raw_parts(ptr, len).to_vec()
    };

    Ok(result)
}

/// Generate uniform random numbers in (0, 1] on Metal GPU
pub fn metalrands(n: usize) -> XResult<Vec<f32>> {
    let device = METAL_DEVICE.as_ref()?;
    let queue = METAL_QUEUE.as_ref()?;
    let library = LIBRARY.as_ref()?;

    let function = library
        .get_function("randuniform", None)
        .map_err(|e| crate::XError::Other(format!("Function not found: {}", e)))?;

    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| crate::XError::Other(format!("Pipeline error: {}", e)))?;

    let out_buffer = device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let seed: u64 = rand::rng().random();
    let len_u32 = n as u32;

    let (thread_groups, threads_per_group) = thread_config(n);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&out_buffer), 0);
    encoder.set_bytes(
        1,
        std::mem::size_of::<u32>() as u64,
        &len_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        2,
        std::mem::size_of::<u64>() as u64,
        &seed as *const u64 as *const _,
    );

    encoder.dispatch_thread_groups(thread_groups, threads_per_group);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result = unsafe {
        let ptr = out_buffer.contents() as *const f32;
        std::slice::from_raw_parts(ptr, n).to_vec()
    };

    Ok(result)
}

/// Generate normal random numbers on Metal GPU
pub fn metalrandn(n: usize, mu: f32, sigma: f32) -> XResult<Vec<f32>> {
    let device = METAL_DEVICE.as_ref()?;
    let queue = METAL_QUEUE.as_ref()?;
    let library = LIBRARY.as_ref()?;

    let function = library
        .get_function("randnormal", None)
        .map_err(|e| crate::XError::Other(format!("Function not found: {}", e)))?;

    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| crate::XError::Other(format!("Pipeline error: {}", e)))?;

    let out_buffer = device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let seed: u64 = rand::rng().random();
    let len_u32 = n as u32;

    let (thread_groups, threads_per_group) = thread_config(n);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&out_buffer), 0);
    encoder.set_bytes(
        1,
        std::mem::size_of::<u32>() as u64,
        &len_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        2,
        std::mem::size_of::<f32>() as u64,
        &mu as *const f32 as *const _,
    );
    encoder.set_bytes(
        3,
        std::mem::size_of::<f32>() as u64,
        &sigma as *const f32 as *const _,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<u64>() as u64,
        &seed as *const u64 as *const _,
    );

    encoder.dispatch_thread_groups(thread_groups, threads_per_group);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result = unsafe {
        let ptr = out_buffer.contents() as *const f32;
        std::slice::from_raw_parts(ptr, n).to_vec()
    };

    Ok(result)
}

/// Generate exponential random numbers on Metal GPU
pub fn metalrandexp(n: usize) -> XResult<Vec<f32>> {
    let device = METAL_DEVICE.as_ref()?;
    let queue = METAL_QUEUE.as_ref()?;
    let library = LIBRARY.as_ref()?;

    let function = library
        .get_function("randexp", None)
        .map_err(|e| crate::XError::Other(format!("Function not found: {}", e)))?;

    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| crate::XError::Other(format!("Pipeline error: {}", e)))?;

    let out_buffer = device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let seed: u64 = rand::rng().random();
    let len_u32 = n as u32;

    let (thread_groups, threads_per_group) = thread_config(n);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&out_buffer), 0);
    encoder.set_bytes(
        1,
        std::mem::size_of::<u32>() as u64,
        &len_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        2,
        std::mem::size_of::<u64>() as u64,
        &seed as *const u64 as *const _,
    );

    encoder.dispatch_thread_groups(thread_groups, threads_per_group);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result = unsafe {
        let ptr = out_buffer.contents() as *const f32;
        std::slice::from_raw_parts(ptr, n).to_vec()
    };

    Ok(result)
}
