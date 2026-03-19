# AGENT.md

This file is guidance for coding agents working in the `diffusionx` repository.

## Project Summary

`diffusionx` is a Rust 2024 library for:

- random number generation
- stochastic process simulation
- optional trajectory visualization
- optional GPU-accelerated moment calculation via CUDA or Metal

The crate is performance-oriented, heavily trait-based, and generic over numeric types where that improves reuse.

## Baseline Facts

- Crate name: `diffusionx`
- Edition: Rust 2024
- MSRV: `1.88`
- Default feature: `mimalloc`
- Optional features: `cuda`, `metal`, `io`, `visualize`
- `cuda` and `metal` are mutually exclusive
- Public fallible APIs return `XResult<T>`
- CI currently runs `cargo build` and `cargo test`

## Repository Map

- `src/lib.rs`: crate root, shared trait bounds, feature-gated modules
- `src/error.rs`: shared error types and `XResult`
- `src/random/`: free-function RNG APIs by distribution
- `src/simulation/basic/`: core process traits and trajectory abstractions
- `src/simulation/continuous/`: continuous-process implementations
- `src/simulation/discrete/`: discrete-process implementations
- `src/simulation/point/`: point-process implementations
- `src/simulation/macros.rs`: helper macros for Langevin-style APIs
- `src/simulation/prelude.rs`: common re-exports
- `src/gpu/`: Rust-side GPU moment implementations
- `src/visualize/`: plotting support behind `visualize`
- `src/utils/`: shared utility functions and optional CSV helpers
- `kernels/cuda-kernel/`: CUDA kernels compiled by `build.rs`
- `kernels/metal-kernel/`: Metal kernels compiled by `build.rs`
- `examples/`: user-facing examples
- `benches/`: Criterion benchmarks

## How To Work In This Repo

Start by reading the nearest module and matching its style. This crate is internally consistent, and changes should extend existing patterns instead of introducing a new abstraction style.

Prefer small, local changes over broad refactors unless the task explicitly requires architectural work.

When adding or changing behavior:

- keep public APIs consistent with existing naming
- preserve generic support when the surrounding module is generic
- use the shared error types instead of ad hoc panics for user-facing failures
- add or update tests in the same file or module area as the implementation
- update docs or examples when the user-visible API changes

## Coding Conventions

### Errors and Validation

- Use `XResult<T>` for fallible public APIs.
- Reuse existing error variants from `src/error.rs`.
- Validate parameters early and return structured errors.
- Reuse shared validation helpers such as `check_duration_time_step` where appropriate.
- Avoid `unwrap()` in library code unless the invariant is internal and already enforced by construction.

### Traits and Generics

- Continuous-process implementations typically use `T: FloatExt`.
- Integer-oriented utilities use `IntExt`; mixed numeric helpers may use `RealExt`.
- New process structs should generally be `Debug + Clone`.
- Implementations should remain `Send + Sync` when practical because simulation and moment code uses parallel execution.

### Randomness and Parallelism

- RNG implementations use `Xoshiro256PlusPlus` as the common entropy source.
- Distribution modules expose free functions such as `rand`, `rands`, `standard_rand`, and `standard_rands`.
- Respect existing sequential/parallel thresholds instead of parallelizing everything unconditionally.
- Parallel work is typically done with `rayon`.

### Process Implementations

For a new process, prefer following the existing structure:

1. define a process struct with validated constructor parameters
2. implement the relevant process trait from `src/simulation/basic/`
3. add focused unit tests in the same file
4. expose the module from its parent `mod.rs`
5. add documentation or example coverage if the API is user-facing

A `ContinuousProcess<T>` implementation usually unlocks trajectory simulation and derived functionality such as moments, FPT, occupation time, TAMSD, and plotting through existing blanket impls and helper traits.

### Documentation

- Keep doc comments concise and aligned with the rest of the crate.
- If you change a public API, check whether `README.md`, examples, or doctext in module docs also needs updating.
- `src/lib.rs` includes `README.md` as crate-level documentation, so README drift affects docs.rs output.

## Feature-Specific Rules

### `visualize`

- Visualization code lives in `src/visualize/`.
- Keep feature gating explicit with `#[cfg(feature = "visualize")]`.
- If a public item depends on plotting support, mirror the crate’s existing `cfg_attr(docsrs, doc(cfg(...)))` style where relevant.

### `io`

- CSV helpers live in `src/utils/csv.rs`.
- Do not introduce unconditional `csv` usage outside the `io` feature path.

### `cuda` and `metal`

- Never enable both features at once. `build.rs` will panic.
- GPU Rust code is split by backend under `src/gpu/cuda/` and `src/gpu/metal/`.
- Kernel source lives under `kernels/`.
- If you change a kernel interface, update all affected layers:
  - Rust-side backend module
  - kernel source
  - any shared loading or build logic in `build.rs`
- CUDA builds require `nvcc`.
- Metal builds require `xcrun metal` and `xcrun metallib` on macOS.

Unless the task is explicitly backend-specific, avoid changing only one GPU backend when the abstraction is meant to stay mirrored across CUDA and Metal.

## Test and Validation Commands

Run the narrowest command that credibly validates your change, then broaden if needed.

Common commands:

```bash
cargo fmt --check
cargo test
cargo test <module_or_test_name>
cargo build --features "visualize,io"
cargo doc --features "visualize,io" --no-deps
cargo bench
```

Feature-specific checks:

```bash
cargo build --features metal
cargo build --features cuda
```

Use GPU feature builds only when the environment has the required toolchain installed.

## Testing Expectations

- Add unit tests next to the implementation you changed.
- Preserve existing test style: lightweight behavioral tests, numerical sanity checks, and `Send + Sync` assertions where useful.
- For numerical algorithms, favor robust invariants and tolerances over brittle exact-value assertions.
- If the change affects examples or public usage patterns, consider validating the relevant example with `cargo run --example <name>`.

## Performance Expectations

This crate is performance-sensitive. Before introducing extra allocation, dynamic dispatch, or synchronization, check whether the local code path is on a hot loop.

Be conservative with:

- repeated heap allocation inside simulation loops
- unnecessary cloning of trajectories or buffers
- replacing iterator patterns that are intentionally straightforward for optimization
- introducing serialization or logging in performance-critical code

## Safe Change Boundaries

Low-risk changes:

- adding tests
- improving docs
- small bug fixes inside a single process implementation
- extending an existing module with consistent APIs

Higher-risk changes that require broader review and validation:

- changing trait signatures in `src/simulation/basic/`
- changing error enums or public return types
- modifying RNG behavior used across distributions
- altering kernel interfaces or GPU trait contracts
- changing README examples that serve as crate-level docs

## Practical Checklist Before Finishing

- code formatted
- relevant tests passed
- feature gates are correct
- public errors use `XResult`
- docs/examples updated if the API changed
- parent `mod.rs` exports updated if a new module was added

If you are unsure how to implement something, copy the nearest existing process or distribution module and adapt it instead of inventing a new pattern.
