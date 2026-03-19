# Project Structure

## Root
- `src/lib.rs` — Crate root. Defines core trait bounds (`FloatExt`, `IntExt`, `RealExt`), sets mimalloc as global allocator, re-exports modules.
- `src/error.rs` — Unified error types (`XError`, `XResult<T>`) using `thiserror`. All fallible functions return `XResult<T>`.
- `build.rs` — Compiles CUDA/Metal GPU kernels at build time.

## `src/random/` — Random Number Generation
Each distribution is a standalone module exposing `rand()`, `rands()`, `standard_rand()`, `standard_rands()` free functions.
- `normal.rs`, `uniform.rs`, `exponential.rs`, `poisson.rs`, `gamma.rs`, `stable.rs`
- Parallelism threshold constants: `PAR_THRESHOLD` (50,000), `STABLE_PAR_THRESHOLD` (1,000)

## `src/simulation/` — Stochastic Process Simulation

### `simulation/basic/` — Core Traits & Abstractions
- `continuous.rs` — `ContinuousProcess<T>` trait + `ContinuousTrajectory` struct
- `discrete.rs` — `DiscreteProcess` trait + `DiscreteTrajectory` struct
- `point.rs` — `PointProcess` trait + `PointTrajectory` struct
- `moment.rs` — `Moment` trait with blanket impls for all trajectory types
- `tamsd.rs` — Time-averaged MSD computation
- `inverse.rs` — Inverse process trait
- `functional.rs` — Functional traits (first passage time, occupation time)

### `simulation/continuous/` — Continuous Process Implementations
Each file implements `ContinuousProcess<T>` for a specific process:
`bm.rs`, `fbm.rs`, `ou.rs`, `levy.rs`, `langevin.rs`, `geometric_bm.rs`, `brownian_bridge.rs`, `brownian_excursion.rs`, `brownian_meander.rs`, `cauchy.rs`, `gamma.rs`, `levy_walk.rs`, `subordinator.rs`, `generalized_langevin.rs`, `bng.rs`

### `simulation/discrete/` — Discrete Process Implementations
- `random_walk.rs`

### `simulation/point/` — Point Process Implementations
- `poisson.rs`, `ctrw.rs`, `birth_death.rs`

### `simulation/macros.rs` — Convenience Macros
`langevin!`, `generalized_langevin!`, `subordinated_langevin!` for SDE construction.

### `simulation/prelude.rs` — Common Re-exports
Imports all basic traits, `FloatExt`, `IntExt`, and conditionally `GPUMoment` and visualization types.

## `src/gpu/` — GPU Acceleration
- `mod.rs` — `GPUMoment` trait definition
- `cuda/` — CUDA implementations (`bm.rs`, `ou.rs`, `levy.rs`, `random.rs`)
- `metal/` — Metal implementations (same structure)

## `kernels/` — GPU Kernel Source
- `cuda-kernel/` — `.cu` files compiled to PTX by build.rs
- `metal-kernel/` — `.metal` files compiled to .metallib by build.rs
- Both include a `utils.cu`/`utils.metal` with shared helpers

## `src/utils/` — Utilities
- `functions.rs` — Math helper functions
- `sgn.rs` — Sign function
- `csv.rs` — CSV I/O (feature-gated on `io`)

## `src/visualize/` — Visualization (feature-gated on `visualize`)
- `config.rs` — `PlotConfig` builder
- `draw.rs` — Plotting logic via plotters

## Other Directories
- `examples/` — Usage examples (`bm.rs`, `CIR.rs`, `langevin.rs`, `randoms.rs`)
- `benches/` — Criterion benchmarks (`bm.rs`, `ou.rs`, `levy.rs`, `random.rs`)

## Architecture Patterns
- Trait-based extensibility: implement `ContinuousProcess<T>` to get moment calculation, FPT, occupation time, TAMSD, and visualization for free.
- Generic over float type `T: FloatExt` (typically `f64`, `f32` for GPU).
- Parallelism via `rayon` with threshold-based switching between sequential and parallel iteration.
- Feature gates for optional capabilities (GPU, visualization, I/O).
- All process structs are `Send + Sync + Clone + Debug`.
- Consistent error handling: all public functions return `XResult<T>`.
