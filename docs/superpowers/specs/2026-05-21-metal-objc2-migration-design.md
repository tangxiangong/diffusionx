# Metal Binding Migration Design

## Context

The crate currently exposes optional GPU acceleration through the `metal` feature and
the `diffusionx::gpu::metal` module. The public API should not change. Users should
continue enabling `features = ["metal"]` and calling the existing `GPUMoment` methods
or `gpu::metal::random::*` functions.

The current backend uses the `metal` crate. That crate is deprecated because the old
`objc` ecosystem is unmaintained, and its documentation recommends new development
use `objc2` and `objc2-metal` instead:
<https://docs.rs/metal/latest/src/metal/lib.rs.html#70-79>.

The replacement crate exposes the Metal Objective-C protocols directly. For example,
`MTLCreateSystemDefaultDevice()` returns a retained Objective-C protocol object, and
the main methods are named after their Objective-C selectors:

- `MTLCreateSystemDefaultDevice`: <https://docs.rs/objc2-metal/latest/objc2_metal/fn.MTLCreateSystemDefaultDevice.html>
- `MTLDevice`: <https://docs.rs/objc2-metal/latest/objc2_metal/trait.MTLDevice.html>
- `MTLCommandQueue`: <https://docs.rs/objc2-metal/latest/objc2_metal/trait.MTLCommandQueue.html>
- `MTLComputeCommandEncoder`: <https://docs.rs/objc2-metal/latest/objc2_metal/trait.MTLComputeCommandEncoder.html>

## Goals

- Replace the unmaintained `metal` crate with `objc2-metal` for the Metal backend.
- Preserve the existing public API, feature name, module paths, and behavior.
- Keep `build.rs` Metal shader compilation unchanged unless migration reveals a direct
  compatibility issue.
- Centralize Objective-C ownership and `unsafe` calls in small private helpers.
- Improve command-buffer error reporting where `objc2-metal` exposes useful status.

## Non-Goals

- Do not rename the Cargo feature from `metal`.
- Do not change `diffusionx::gpu::metal` public functions or `GPUMoment`.
- Do not rewrite Metal kernels.
- Do not redesign the CUDA backend.
- Do not introduce a cross-backend GPU abstraction as part of this migration.

## Dependency Plan

`Cargo.toml` should keep:

```toml
[features]
metal = ["dep:objc2-metal"]
```

The implementation may also need a direct `objc2` and/or `objc2-foundation`
dependency if the types are not conveniently re-exported by `objc2-metal`.
These dependencies must remain optional and enabled only by the `metal` feature.

The old dependency:

```toml
metal = { version = "0.33", optional = true }
```

should be removed once the backend no longer imports it.

## Migration Boundary

The migration is limited to the `#[cfg(feature = "metal")]` backend:

- `src/gpu/metal/mod.rs`
- `src/gpu/metal/random.rs`
- Type references in `src/gpu/metal/{bm,ou,levy}.rs`
- `Cargo.toml` and `Cargo.lock`

The following should remain behaviorally unchanged:

- `src/gpu/mod.rs`
- `src/simulation/prelude.rs`
- `build.rs` Metal branch
- `kernels/metal-kernel/*.metal`
- README examples and benchmark call sites

## Internal Adapter Design

`src/gpu/metal/mod.rs` should own a private adapter layer around `objc2-metal`.
The rest of the Metal backend should not need to manipulate Objective-C ownership
or raw pointers directly.

Suggested private type aliases:

```rust
type MetalDevice = objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>;
type MetalQueue = objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>;
type MetalLibrary = objc2::runtime::ProtocolObject<dyn objc2_metal::MTLLibrary>;
type MetalPipeline = objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>;
type MetalBuffer = objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>;
```

The exact aliases may be adjusted during implementation to match the crate's latest
type paths, but the adapter should store retained Objective-C objects, not borrowed
objects with unclear lifetimes.

Core helpers:

- `METAL_DEVICE: LazyLock<XResult<Retained<MetalDevice>>>`
  - Uses `MTLCreateSystemDefaultDevice()`.
  - Returns `XError::Other("No Metal device found")` if unavailable.
- `METAL_QUEUE: LazyLock<XResult<Retained<MetalQueue>>>`
  - Uses `device.newCommandQueue()`.
  - Converts a `None` return into `XError::Other`.
- `load_library(path: &str) -> XResult<Retained<MetalLibrary>>`
  - Converts `path` through `NSString::from_str`.
  - Uses `device.newLibraryWithFile_error(...)`.
  - Includes the path and `NSError` description in failures.
- `get_pipeline(library, function_name) -> XResult<Retained<MetalPipeline>>`
  - Converts `function_name` through `NSString::from_str`.
  - Uses `library.newFunctionWithName(...)`.
  - Uses `device.newComputePipelineStateWithFunction_error(...)`.
  - Includes the kernel name and `NSError` description in failures.
- `thread_config(particles: usize) -> (MTLSize, MTLSize)`
  - Keeps the current 256-thread group size and `particles.div_ceil(256)`.

Buffer and encoder helpers:

- `new_shared_buffer(bytes: usize) -> XResult<Retained<MetalBuffer>>`
  - Uses `MTLResourceOptions::StorageModeShared`.
  - Treats `None` as allocation failure.
- `zero_buffer_f32(buffer)` and `read_buffer_f32(buffer)`
  - Use `MTLBuffer::contents()` and pointer casts in one place.
- `read_buffer_vec_f32(buffer, len)`
  - Reads a completed shared buffer into `Vec<f32>`.
- `set_buffer(encoder, index, buffer)`
  - Wraps unsafe `setBuffer_offset_atIndex`.
- `set_scalar<T>(encoder, index, &value)`
  - Wraps unsafe `setBytes_length_atIndex`.
- `finish_command_buffer(command_buffer)`
  - Calls `commit()` and `waitUntilCompleted()`.
  - Checks `command_buffer.error()` after completion and maps it to `XError::Other`.

## Data Flow

Moment kernels keep the existing execution flow:

1. Lazily load the `.metallib` path injected by `build.rs`.
2. Lazily create one pipeline per kernel function.
3. Allocate one shared `f32` output buffer.
4. Zero-initialize the output buffer.
5. Generate a host seed with `rand`.
6. Create a command buffer and compute encoder.
7. Bind output buffer at index 0.
8. Bind scalar kernel parameters with `set_scalar`.
9. Bind `particles_u32` and seed.
10. Bind threadgroup memory for moment reductions.
11. Dispatch thread groups.
12. End encoding, wait for completion, check command-buffer error.
13. Read the output sum and return `sum / particles as f32`.

Central moment kernels keep their two-pass flow: compute mean first, then run the
central moment kernel using that mean.

Random kernels keep the existing execution flow:

1. Allocate one shared output buffer sized for `len` or `n` `f32` values.
2. Bind output buffer at index 0.
3. Bind scalar distribution parameters, length, and seed.
4. Dispatch and wait.
5. Read a `Vec<f32>` of the requested length.

## Error Handling

Public fallible APIs continue returning `XResult<T>`.

New or preserved error cases should include:

- No system Metal device.
- Command queue creation failed.
- Library path could not be loaded.
- Kernel function not found in library.
- Compute pipeline creation failed.
- Buffer allocation failed.
- Command buffer creation failed.
- Compute encoder creation failed.
- Command buffer completed with an error.

Errors from Objective-C APIs should include the operation name and, when practical,
the `NSError` localized description.

## Safety Rules

All `unsafe` calls introduced by `objc2-metal` should stay inside private helpers
or tiny local blocks in `src/gpu/metal/mod.rs`.

Each unsafe block should be justified by a short comment covering:

- The pointer comes from a live stack value or retained Metal buffer.
- The value length matches the Metal kernel argument type.
- Bound buffers remain alive until `waitUntilCompleted()` returns.
- Shared-buffer reads happen only after command-buffer completion.

The process-specific modules should not contain raw Objective-C message calls or
manual `NonNull<c_void>` construction unless a later implementation proves a helper
would obscure correctness.

## Testing and Verification

Start with non-Metal checks to ensure the optional dependency change did not affect
default users:

```sh
cargo check --all
cargo test --lib
```

Then verify the Metal feature on macOS with the Apple Metal toolchain:

```sh
cargo check --features metal --no-default-features
cargo test --features metal --no-default-features gpu::metal
```

Run formatting and linting before completion:

```sh
cargo fmt -- --check
cargo clippy --all-targets --tests --benches -- -D warnings
```

If Metal-specific clippy is supported in the local environment, run:

```sh
cargo clippy --all-targets --features metal --no-default-features --tests --benches -- -D warnings
```

Expected behavior checks:

- Existing BM and OU Metal moment tests pass.
- Metal random functions return the requested number of values.
- Uniform RNG values remain in the expected range.
- Normal, exponential, and stable RNG outputs are finite for small deterministic
  smoke-test sizes.
- README GPU examples and Metal benchmark call sites still compile without API edits.

## Implementation Sequence

1. Replace optional dependency metadata in `Cargo.toml`.
2. Introduce private objc2-metal type aliases and helpers in `src/gpu/metal/mod.rs`.
3. Port `METAL_DEVICE`, `METAL_QUEUE`, `load_library`, `get_pipeline`, and
   `thread_config`.
4. Port the two Metal moment macros to call the new helpers.
5. Port `src/gpu/metal/random.rs` to the new helpers.
6. Adjust `bm.rs`, `ou.rs`, and `levy.rs` only where concrete `metal::...` type
   names need replacement.
7. Run narrow Metal build checks, then non-Metal checks, then formatting and linting.
8. Update docs only if the public behavior or setup requirements changed.

## Open Decisions

No product/API decisions are open. During implementation, exact dependency feature
flags should be selected by the compiler errors and the `objc2-metal` feature list,
but they must remain optional and gated behind the existing `metal` feature.
