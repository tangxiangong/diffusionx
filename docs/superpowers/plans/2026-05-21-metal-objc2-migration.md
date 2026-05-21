# Metal objc2 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the deprecated `metal` crate with `objc2-metal` while preserving the existing `metal` Cargo feature and public GPU API.

**Architecture:** Keep the public `diffusionx::gpu::metal` module intact and add a private adapter layer inside `src/gpu/metal/mod.rs`. The adapter owns Objective-C retained objects, `NSString` conversion, buffer allocation, scalar binding, command completion, and all new `unsafe` calls so process modules stay focused on kernel semantics.

**Tech Stack:** Rust 2024, `objc2`, `objc2-foundation`, `objc2-metal`, Apple Metal command APIs, existing `xcrun metal` / `xcrun metallib` build flow.

---

## File Structure

- Modify `Cargo.toml`
  - Remove the optional `metal` crate dependency.
  - Add optional `objc2`, `objc2-foundation`, and `objc2-metal` dependencies behind the existing `metal` feature.
- Modify `Cargo.lock`
  - Let Cargo update lockfile entries after dependency replacement.
- Modify `src/gpu/metal/mod.rs`
  - Replace all direct `metal` crate imports.
  - Add private objc2-metal type aliases and helper functions.
  - Port `METAL_DEVICE`, `METAL_QUEUE`, `load_library`, `get_pipeline`, `thread_config`, and the two Metal macros.
- Modify `src/gpu/metal/random.rs`
  - Replace direct `metal` crate use with the new helpers.
  - Keep public function signatures unchanged.
- Modify `src/gpu/metal/{bm.rs,ou.rs,levy.rs}`
  - Only adjust concrete library/pipeline type names if needed.
- Optional modify `README.md`
  - Only if implementation changes setup requirements. The expected path is no README change.

## Task 1: Dependency Gate

**Files:**
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`

- [ ] **Step 1: Update optional dependency metadata**

Edit `Cargo.toml` so the feature and dependency section use:

```toml
[features]
cuda = ["dep:cudarc"]
default = ["mimalloc"]
io = ["dep:csv"]
metal = ["dep:objc2", "dep:objc2-foundation", "dep:objc2-metal"]
mimalloc = ["dep:mimalloc"]
visualize = [
    "dep:derive_builder",
    "dep:either",
    "dep:plotters",
    "dep:plotters-backend",
]
```

Replace:

```toml
metal = { version = "0.33", optional = true }
```

with:

```toml
objc2 = { version = "0.6", optional = true }
objc2-foundation = { version = "0.3", optional = true, features = ["NSString"] }
objc2-metal = { version = "0.3", optional = true }
```

- [ ] **Step 2: Run metadata check to refresh the lockfile**

Run:

```sh
cargo check --no-default-features
```

Expected: PASS. This should not compile the Metal backend, but it should update `Cargo.lock` if needed.

- [ ] **Step 3: Verify no public feature rename happened**

Run:

```sh
rg -n 'metal = \[' Cargo.toml
rg -n 'dep:metal|metal = \{ version' Cargo.toml
```

Expected: first command prints the `metal = ["dep:objc2", ...]` feature; second command prints nothing.

- [ ] **Step 4: Commit dependency metadata**

Run:

```sh
git add Cargo.toml Cargo.lock
printf '%s\n' \
'build(gpu): replace deprecated Metal binding dependency' \
'' \
'Summary:' \
'- remove the optional deprecated metal crate dependency' \
'- add optional objc2, objc2-foundation, and objc2-metal dependencies behind the existing metal feature' \
'' \
'Rationale:' \
'- prepare the Metal backend migration without changing the public feature name' \
'- keep replacement bindings optional so default builds remain unaffected' \
'' \
'Tests:' \
'- cargo check --no-default-features' \
'' \
'Co-authored-by: Codex <noreply@openai.com>' \
> /private/tmp/diffusionx-commit.txt
git commit -F /private/tmp/diffusionx-commit.txt
```

Commit message content:

```text
build(gpu): replace deprecated Metal binding dependency

Summary:
- remove the optional deprecated metal crate dependency
- add optional objc2, objc2-foundation, and objc2-metal dependencies behind the existing metal feature

Rationale:
- prepare the Metal backend migration without changing the public feature name
- keep replacement bindings optional so default builds remain unaffected

Tests:
- cargo check --no-default-features

Co-authored-by: Codex <noreply@openai.com>
```

## Task 2: Adapter Types and Device Setup

**Files:**
- Modify: `src/gpu/metal/mod.rs`

- [ ] **Step 1: Replace imports and add type aliases**

At the top of `src/gpu/metal/mod.rs`, replace the current Metal imports with:

```rust
use crate::{XError, XResult};
use objc2::{
    rc::Retained,
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLFunction, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use std::{ffi::c_void, ptr::NonNull, sync::LazyLock};

type MetalDevice = ProtocolObject<dyn MTLDevice>;
type MetalQueue = ProtocolObject<dyn MTLCommandQueue>;
type MetalLibrary = ProtocolObject<dyn MTLLibrary>;
type MetalFunction = ProtocolObject<dyn MTLFunction>;
type MetalPipeline = ProtocolObject<dyn MTLComputePipelineState>;
type MetalBuffer = ProtocolObject<dyn MTLBuffer>;
type MetalCommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;
type MetalComputeEncoder = ProtocolObject<dyn MTLComputeCommandEncoder>;
```

- [ ] **Step 2: Port device and queue statics**

Replace the current `METAL_DEVICE` and `METAL_QUEUE` definitions with:

```rust
pub(crate) static METAL_DEVICE: LazyLock<XResult<Retained<MetalDevice>>> = LazyLock::new(|| {
    MTLCreateSystemDefaultDevice()
        .ok_or_else(|| XError::Other("No Metal device found".into()))
});

pub(crate) static METAL_QUEUE: LazyLock<XResult<Retained<MetalQueue>>> = LazyLock::new(|| {
    let device = METAL_DEVICE.as_ref()?;
    device
        .newCommandQueue()
        .ok_or_else(|| XError::Other("Failed to create Metal command queue".into()))
});
```

- [ ] **Step 3: Port thread configuration**

Replace `thread_config` with:

```rust
#[inline]
pub(crate) fn thread_config(particles: usize) -> (MTLSize, MTLSize) {
    let thread_group_size = 256usize;
    let thread_groups = particles.div_ceil(thread_group_size);

    (
        MTLSize {
            width: thread_groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: thread_group_size,
            height: 1,
            depth: 1,
        },
    )
}
```

- [ ] **Step 4: Compile to expose exact objc2 API mismatches**

Run:

```sh
cargo check --features metal --no-default-features
```

Expected: FAIL at remaining old `metal::...` type references and unported methods. If it fails on `MTLSize` integer type, replace `usize` fields with `thread_groups as _` and `thread_group_size as _`.

## Task 3: Library, Pipeline, Buffer, and Command Helpers

**Files:**
- Modify: `src/gpu/metal/mod.rs`

- [ ] **Step 1: Add NSError formatting helper**

Add below the `.metallib` constants:

```rust
fn ns_error_message(error: &objc2_foundation::NSError) -> String {
    error.localizedDescription().to_string()
}
```

If `NSError` is not available with the current feature set, update the dependency to:

```toml
objc2-foundation = { version = "0.3", optional = true, features = ["NSError", "NSString"] }
```

- [ ] **Step 2: Port `load_library`**

Replace the current function with:

```rust
pub(crate) fn load_library(path: &str) -> XResult<Retained<MetalLibrary>> {
    let device = METAL_DEVICE.as_ref()?;
    let path = NSString::from_str(path);

    device
        .newLibraryWithFile_error(&path)
        .map_err(|error| {
            XError::Other(format!(
                "Failed to load Metal library '{}': {}",
                path,
                ns_error_message(&error)
            ))
        })
}
```

If `NSString` does not format cleanly with `{}`, use `path.to_string()` before the call and include that Rust string in the error.

- [ ] **Step 3: Port `get_pipeline`**

Replace the current function with:

```rust
pub(crate) fn get_pipeline(
    library: &MetalLibrary,
    function_name: &str,
) -> XResult<Retained<MetalPipeline>> {
    let device = METAL_DEVICE.as_ref()?;
    let function_name_ns = NSString::from_str(function_name);
    let function: Retained<MetalFunction> = library
        .newFunctionWithName(&function_name_ns)
        .ok_or_else(|| XError::Other(format!("Function '{function_name}' not found")))?;

    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| {
            XError::Other(format!(
                "Pipeline creation error for '{function_name}': {}",
                ns_error_message(&error)
            ))
        })
}
```

- [ ] **Step 4: Add buffer and encoder helpers**

Add the following helpers before the macros:

```rust
pub(crate) fn new_shared_buffer(bytes: usize) -> XResult<Retained<MetalBuffer>> {
    let device = METAL_DEVICE.as_ref()?;
    device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| XError::Other(format!("Failed to allocate Metal shared buffer: {bytes} bytes")))
}

pub(crate) fn zero_buffer_f32(buffer: &MetalBuffer) {
    // SAFETY: `contents` is valid for a shared Metal buffer, and this writes one f32
    // before the buffer is submitted to the GPU.
    unsafe {
        let ptr = buffer.contents().as_ptr().cast::<f32>();
        ptr.write(0.0);
    }
}

pub(crate) fn read_buffer_f32(buffer: &MetalBuffer) -> f32 {
    // SAFETY: command-buffer completion is awaited before callers read the shared buffer.
    unsafe { buffer.contents().as_ptr().cast::<f32>().read() }
}

pub(crate) fn read_buffer_vec_f32(buffer: &MetalBuffer, len: usize) -> Vec<f32> {
    // SAFETY: command-buffer completion is awaited before callers read `len` f32 values.
    unsafe {
        let ptr = buffer.contents().as_ptr().cast::<f32>();
        std::slice::from_raw_parts(ptr, len).to_vec()
    }
}

pub(crate) fn set_buffer(
    encoder: &MetalComputeEncoder,
    index: usize,
    buffer: &MetalBuffer,
) {
    // SAFETY: the retained buffer lives until after `waitUntilCompleted`, offset is zero,
    // and the binding index matches the Metal kernel signature.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
    }
}

pub(crate) fn set_scalar<T>(
    encoder: &MetalComputeEncoder,
    index: usize,
    value: &T,
) {
    let ptr = NonNull::from(value).cast::<c_void>();
    // SAFETY: Metal copies `size_of::<T>()` bytes immediately from a valid stack value,
    // and scalar bindings are smaller than Metal's inline setBytes limit.
    unsafe {
        encoder.setBytes_length_atIndex(ptr, std::mem::size_of::<T>(), index);
    }
}

pub(crate) fn finish_command_buffer(command_buffer: &MetalCommandBuffer) -> XResult<()> {
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    if let Some(error) = command_buffer.error() {
        return Err(XError::Other(format!(
            "Metal command buffer failed: {}",
            ns_error_message(&error)
        )));
    }

    Ok(())
}
```

If objc2 expects `NSUInteger` rather than `usize`, cast every index and length argument with `as _` at the call site inside these helpers.

- [ ] **Step 5: Compile adapter helpers**

Run:

```sh
cargo check --features metal --no-default-features
```

Expected: FAIL only in macro/random code still using old method names. Fix helper signatures if the compiler reports exact reference type mismatches, keeping the helper names and responsibilities intact.

## Task 4: Port Moment Macros

**Files:**
- Modify: `src/gpu/metal/mod.rs`

- [ ] **Step 1: Replace the regular moment macro body**

Inside `subscribe_metal_gpu_function!`, keep the generated function signature unchanged and replace the body with:

```rust
let queue = $crate::gpu::metal::METAL_QUEUE.as_ref()?;
static PIPELINE: std::sync::LazyLock<XResult<Retained<MetalPipeline>>> =
    std::sync::LazyLock::new(|| {
        let library = $library.as_ref()?;
        $crate::gpu::metal::get_pipeline(library, $kernel_name)
    });
let pipeline = PIPELINE.as_ref()?;

let (thread_groups, threads_per_group) = $crate::gpu::metal::thread_config(particles);
let out_buffer = $crate::gpu::metal::new_shared_buffer(std::mem::size_of::<f32>())?;
$crate::gpu::metal::zero_buffer_f32(&out_buffer);

let mut rng = rand::rng();
use rand::RngExt;
let seed: u64 = rng.random();
let particles_u32 = particles as u32;

let command_buffer = queue
    .commandBuffer()
    .ok_or_else(|| $crate::XError::Other("Failed to create Metal command buffer".into()))?;
let encoder = command_buffer
    .computeCommandEncoder()
    .ok_or_else(|| $crate::XError::Other("Failed to create Metal compute encoder".into()))?;

encoder.setComputePipelineState(pipeline);

let mut buffer_index = 0usize;
$crate::gpu::metal::set_buffer(&encoder, buffer_index, &out_buffer);
buffer_index += 1;

$(
    $crate::gpu::metal::set_scalar(&encoder, buffer_index, &$param_name);
    buffer_index += 1;
)+

$crate::gpu::metal::set_scalar(&encoder, buffer_index, &particles_u32);
buffer_index += 1;
$crate::gpu::metal::set_scalar(&encoder, buffer_index, &seed);

encoder.setThreadgroupMemoryLength_atIndex(32 * std::mem::size_of::<f32>(), 0);
encoder.dispatchThreadgroups_threadsPerThreadgroup(thread_groups, threads_per_group);
encoder.endEncoding();

$crate::gpu::metal::finish_command_buffer(&command_buffer)?;

let sum = $crate::gpu::metal::read_buffer_f32(&out_buffer);
Ok(sum / particles as f32)
```

If the `Retained<MetalPipeline>` type alias is not in macro scope, qualify it as
`objc2::rc::Retained<$crate::gpu::metal::MetalPipeline>` and make the alias `pub(crate)`.

- [ ] **Step 2: Replace the central moment macro body**

Inside `subscribe_metal_central_moment_gpu_function!`, keep the generated function signature unchanged and replace the body with:

```rust
let queue = $crate::gpu::metal::METAL_QUEUE.as_ref()?;
static PIPELINE: std::sync::LazyLock<XResult<Retained<MetalPipeline>>> =
    std::sync::LazyLock::new(|| {
        let library = $library.as_ref()?;
        $crate::gpu::metal::get_pipeline(library, $kernel_name)
    });
let pipeline = PIPELINE.as_ref()?;

let (thread_groups, threads_per_group) = $crate::gpu::metal::thread_config(particles);
let mean_val = mean($($param_name,)+ particles)?;
let out_buffer = $crate::gpu::metal::new_shared_buffer(std::mem::size_of::<f32>())?;
$crate::gpu::metal::zero_buffer_f32(&out_buffer);

let mut rng = rand::rng();
use rand::RngExt;
let seed: u64 = rng.random();
let particles_u32 = particles as u32;

let command_buffer = queue
    .commandBuffer()
    .ok_or_else(|| $crate::XError::Other("Failed to create Metal command buffer".into()))?;
let encoder = command_buffer
    .computeCommandEncoder()
    .ok_or_else(|| $crate::XError::Other("Failed to create Metal compute encoder".into()))?;

encoder.setComputePipelineState(pipeline);

let mut buffer_index = 0usize;
$crate::gpu::metal::set_buffer(&encoder, buffer_index, &out_buffer);
buffer_index += 1;
$crate::gpu::metal::set_scalar(&encoder, buffer_index, &order);
buffer_index += 1;
$crate::gpu::metal::set_scalar(&encoder, buffer_index, &mean_val);
buffer_index += 1;

$(
    $crate::gpu::metal::set_scalar(&encoder, buffer_index, &$param_name);
    buffer_index += 1;
)+

$crate::gpu::metal::set_scalar(&encoder, buffer_index, &particles_u32);
buffer_index += 1;
$crate::gpu::metal::set_scalar(&encoder, buffer_index, &seed);

encoder.setThreadgroupMemoryLength_atIndex(32 * std::mem::size_of::<f32>(), 0);
encoder.dispatchThreadgroups_threadsPerThreadgroup(thread_groups, threads_per_group);
encoder.endEncoding();

$crate::gpu::metal::finish_command_buffer(&command_buffer)?;

let sum = $crate::gpu::metal::read_buffer_f32(&out_buffer);
Ok(sum / particles as f32)
```

- [ ] **Step 3: Run Metal compile check**

Run:

```sh
cargo check --features metal --no-default-features
```

Expected: FAIL only in `src/gpu/metal/random.rs` and process files with old `metal::Library` / `metal::ComputePipelineState` names.

- [ ] **Step 4: Commit macro migration**

Run:

```sh
git add src/gpu/metal/mod.rs
git commit -F /private/tmp/diffusionx-commit.txt
```

Use this message:

```text
refactor(gpu): port Metal moment dispatch to objc2

Summary:
- replace Metal device, queue, library, pipeline, buffer, and encoder calls with objc2-metal helpers
- keep moment macro signatures and generated GPU estimator behavior unchanged

Rationale:
- move Objective-C ownership and unsafe encoder binding into one private adapter layer
- preserve the public metal feature API while replacing the deprecated binding crate

Tests:
- cargo check --features metal --no-default-features: expected remaining random/backend type errors before random port

Co-authored-by: Codex <noreply@openai.com>
```

## Task 5: Port Metal Random Functions

**Files:**
- Modify: `src/gpu/metal/random.rs`

- [ ] **Step 1: Replace imports and static types**

At the top of `src/gpu/metal/random.rs`, use:

```rust
use crate::{
    XError, XResult,
    gpu::metal::{
        MetalLibrary, MetalPipeline, RANDOM_METALLIB, finish_command_buffer, get_pipeline,
        load_library, new_shared_buffer, read_buffer_vec_f32, set_buffer, set_scalar,
        thread_config,
    },
};
use objc2::rc::Retained;
use rand::RngExt;
use std::sync::LazyLock;

static LIBRARY: LazyLock<XResult<Retained<MetalLibrary>>> =
    LazyLock::new(|| load_library(RANDOM_METALLIB));
static STANDARD_STABLE_PIPELINE: LazyLock<XResult<Retained<MetalPipeline>>> =
    LazyLock::new(|| {
        let library = LIBRARY.as_ref()?;
        get_pipeline(library, "standard_stable_rand")
    });
static UNIFORM_PIPELINE: LazyLock<XResult<Retained<MetalPipeline>>> = LazyLock::new(|| {
    let library = LIBRARY.as_ref()?;
    get_pipeline(library, "randuniform")
});
static NORMAL_PIPELINE: LazyLock<XResult<Retained<MetalPipeline>>> = LazyLock::new(|| {
    let library = LIBRARY.as_ref()?;
    get_pipeline(library, "randnormal")
});
static EXP_PIPELINE: LazyLock<XResult<Retained<MetalPipeline>>> = LazyLock::new(|| {
    let library = LIBRARY.as_ref()?;
    get_pipeline(library, "randexp")
});
```

Make `MetalLibrary` and `MetalPipeline` aliases `pub(crate)` in `mod.rs`.

- [ ] **Step 2: Add shared kernel launch helper**

Add below the static pipelines:

```rust
fn command_encoder(
    pipeline: &MetalPipeline,
) -> XResult<(
    Retained<crate::gpu::metal::MetalCommandBuffer>,
    Retained<crate::gpu::metal::MetalComputeEncoder>,
)> {
    let queue = crate::gpu::metal::METAL_QUEUE.as_ref()?;
    let command_buffer = queue
        .commandBuffer()
        .ok_or_else(|| XError::Other("Failed to create Metal command buffer".into()))?;
    let encoder = command_buffer
        .computeCommandEncoder()
        .ok_or_else(|| XError::Other("Failed to create Metal compute encoder".into()))?;
    encoder.setComputePipelineState(pipeline);
    Ok((command_buffer, encoder))
}
```

Make `MetalCommandBuffer` and `MetalComputeEncoder` aliases `pub(crate)` in `mod.rs`.

- [ ] **Step 3: Port `standard_stable_rands` body**

Replace only the Metal object setup, binding, dispatch, and readback with:

```rust
let pipeline = STANDARD_STABLE_PIPELINE.as_ref()?;

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
let (command_buffer, encoder) = command_encoder(pipeline)?;

set_buffer(&encoder, 0, &out_buffer);
set_scalar(&encoder, 1, &alpha);
set_scalar(&encoder, 2, &beta);
set_scalar(&encoder, 3, &inv_alpha);
set_scalar(&encoder, 4, &one_minus_alpha_div_alpha);
set_scalar(&encoder, 5, &b);
set_scalar(&encoder, 6, &s);
set_scalar(&encoder, 7, &len_u32);
set_scalar(&encoder, 8, &seed);

encoder.dispatchThreadgroups_threadsPerThreadgroup(thread_groups, threads_per_group);
encoder.endEncoding();
finish_command_buffer(&command_buffer)?;

Ok(read_buffer_vec_f32(&out_buffer, len))
```

- [ ] **Step 4: Port `metalrands` body**

Use:

```rust
let pipeline = UNIFORM_PIPELINE.as_ref()?;
let out_buffer = new_shared_buffer(n * std::mem::size_of::<f32>())?;
let seed: u64 = rand::rng().random();
let len_u32 = n as u32;
let (thread_groups, threads_per_group) = thread_config(n);
let (command_buffer, encoder) = command_encoder(pipeline)?;

set_buffer(&encoder, 0, &out_buffer);
set_scalar(&encoder, 1, &len_u32);
set_scalar(&encoder, 2, &seed);

encoder.dispatchThreadgroups_threadsPerThreadgroup(thread_groups, threads_per_group);
encoder.endEncoding();
finish_command_buffer(&command_buffer)?;

Ok(read_buffer_vec_f32(&out_buffer, n))
```

- [ ] **Step 5: Port `metalrandn` body**

Use:

```rust
let pipeline = NORMAL_PIPELINE.as_ref()?;
let out_buffer = new_shared_buffer(n * std::mem::size_of::<f32>())?;
let seed: u64 = rand::rng().random();
let len_u32 = n as u32;
let (thread_groups, threads_per_group) = thread_config(n);
let (command_buffer, encoder) = command_encoder(pipeline)?;

set_buffer(&encoder, 0, &out_buffer);
set_scalar(&encoder, 1, &len_u32);
set_scalar(&encoder, 2, &mu);
set_scalar(&encoder, 3, &sigma);
set_scalar(&encoder, 4, &seed);

encoder.dispatchThreadgroups_threadsPerThreadgroup(thread_groups, threads_per_group);
encoder.endEncoding();
finish_command_buffer(&command_buffer)?;

Ok(read_buffer_vec_f32(&out_buffer, n))
```

- [ ] **Step 6: Port `metalrandexp` body**

Use:

```rust
let pipeline = EXP_PIPELINE.as_ref()?;
let out_buffer = new_shared_buffer(n * std::mem::size_of::<f32>())?;
let seed: u64 = rand::rng().random();
let len_u32 = n as u32;
let (thread_groups, threads_per_group) = thread_config(n);
let (command_buffer, encoder) = command_encoder(pipeline)?;

set_buffer(&encoder, 0, &out_buffer);
set_scalar(&encoder, 1, &len_u32);
set_scalar(&encoder, 2, &seed);

encoder.dispatchThreadgroups_threadsPerThreadgroup(thread_groups, threads_per_group);
encoder.endEncoding();
finish_command_buffer(&command_buffer)?;

Ok(read_buffer_vec_f32(&out_buffer, n))
```

- [ ] **Step 7: Compile Metal backend**

Run:

```sh
cargo check --features metal --no-default-features
```

Expected: PASS or only process-file concrete type errors. Fix process-file type errors by replacing `metal::Library` with `Retained<MetalLibrary>` imports and leaving public APIs untouched.

- [ ] **Step 8: Commit random migration**

Run:

```sh
git add src/gpu/metal/random.rs src/gpu/metal/bm.rs src/gpu/metal/ou.rs src/gpu/metal/levy.rs src/gpu/metal/mod.rs
git commit -F /private/tmp/diffusionx-commit.txt
```

Use this message:

```text
refactor(gpu): port Metal random kernels to objc2

Summary:
- move Metal random kernel dispatch and readback to the objc2-metal adapter helpers
- preserve existing random function signatures and output semantics

Rationale:
- complete removal of direct deprecated metal crate usage from the Metal backend
- keep unsafe buffer binding centralized in the backend adapter

Tests:
- cargo check --features metal --no-default-features

Co-authored-by: Codex <noreply@openai.com>
```

## Task 6: Add Metal Smoke Tests

**Files:**
- Modify: `src/gpu/metal/random.rs`

- [ ] **Step 1: Add random smoke tests**

Append this test module to `src/gpu/metal/random.rs`:

```rust
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
```

- [ ] **Step 2: Run targeted Metal random tests**

Run:

```sh
cargo test --features metal --no-default-features gpu::metal::random
```

Expected: PASS on a macOS machine with Metal support and Xcode command-line tools.

- [ ] **Step 3: Run existing Metal moment tests**

Run:

```sh
cargo test --features metal --no-default-features gpu::metal::bm::tests::test_gpu_moment
cargo test --features metal --no-default-features gpu::metal::ou::tests::test_gpu_moment
```

Expected: PASS. If Lévy has no explicit test, do not add a broad stochastic accuracy test in this migration; keep the new coverage focused on porting the binding layer.

- [ ] **Step 4: Commit smoke tests**

Run:

```sh
git add src/gpu/metal/random.rs
git commit -F /private/tmp/diffusionx-commit.txt
```

Use this message:

```text
test(gpu): add Metal random smoke coverage

Summary:
- add Metal random backend smoke tests for length, finite values, and basic ranges

Rationale:
- verify objc2-metal readback behavior for vector-producing kernels
- keep stochastic assertions broad enough to avoid brittle random-output failures

Tests:
- cargo test --features metal --no-default-features gpu::metal::random
- cargo test --features metal --no-default-features gpu::metal::bm::tests::test_gpu_moment
- cargo test --features metal --no-default-features gpu::metal::ou::tests::test_gpu_moment

Co-authored-by: Codex <noreply@openai.com>
```

## Task 7: Full Verification and Cleanup

**Files:**
- Inspect: all modified files
- Optional Modify: `README.md`

- [ ] **Step 1: Search for deprecated crate usage**

Run:

```sh
rg -n 'use metal|metal::|MTLResourceOptions|new_buffer|new_command_buffer|new_compute_command_encoder|dispatch_thread_groups|set_bytes|set_buffer' src/gpu/metal Cargo.toml
```

Expected: no old `metal::` crate references. `MTLResourceOptions` may appear only from `objc2_metal`.

- [ ] **Step 2: Run non-Metal verification**

Run:

```sh
cargo check --all
cargo test --lib
```

Expected: PASS. These commands protect default users who do not enable Metal.

- [ ] **Step 3: Run Metal verification**

Run:

```sh
cargo check --features metal --no-default-features
cargo test --features metal --no-default-features gpu::metal
```

Expected: PASS on a Metal-capable macOS system with Xcode command-line tools.

- [ ] **Step 4: Run formatting and linting**

Run:

```sh
cargo fmt -- --check
cargo clippy --all-targets --tests --benches -- -D warnings
cargo clippy --all-targets --features metal --no-default-features --tests --benches -- -D warnings
```

Expected: PASS. If Metal clippy cannot run because the machine lacks Metal compiler tools, record the exact error in the final response and ensure non-Metal clippy passes.

- [ ] **Step 5: Decide whether README changed**

Run:

```sh
git diff -- README.md
```

Expected: no README diff. If dependency migration changed user-facing setup instructions, add a short README note under GPU Acceleration that Metal builds use `objc2-metal` internally but still enable `features = ["metal"]`.

- [ ] **Step 6: Final commit for cleanup if needed**

If formatting, README, or small follow-up cleanup changed files after prior commits, commit only those changes:

```text
chore(gpu): finish Metal objc2 migration cleanup

Summary:
- apply final formatting and documentation cleanup for the Metal objc2 migration

Rationale:
- keep the migration branch clean after full verification

Tests:
- cargo check --all
- cargo test --lib
- cargo check --features metal --no-default-features
- cargo test --features metal --no-default-features gpu::metal
- cargo fmt -- --check
- cargo clippy --all-targets --tests --benches -- -D warnings
- cargo clippy --all-targets --features metal --no-default-features --tests --benches -- -D warnings

Co-authored-by: Codex <noreply@openai.com>
```

Do not create an empty cleanup commit if there are no changes.

## Self-Review

- Spec coverage: The plan preserves the `metal` feature and public API, replaces the deprecated binding dependency, keeps `build.rs` and kernels unchanged by default, centralizes Objective-C ownership and unsafe calls, adds command-buffer error checks, and includes Metal/non-Metal verification.
- Placeholder scan: No unresolved placeholder markers or open-ended test-writing steps remain; every code-changing task includes concrete snippets and commands.
- Type consistency: Helper names introduced in Task 3 are reused in Task 4 and Task 5. Public APIs named in the spec remain unchanged.
