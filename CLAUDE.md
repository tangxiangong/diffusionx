# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Companion file

`AGENTS.md` holds the exhaustive command list and the project's style/design conventions
(imports, naming, error handling, parallelism, feature gating, testing expectations).
Read it for any non-trivial change. This file covers the essentials and the big-picture
architecture.

## Common commands

```sh
cargo build                          # default features (mimalloc only)
cargo check --all
cargo clippy --all-targets --tests --benches -- -D warnings
cargo fmt -- --check
cargo test                           # library tests; doctests are disabled
cargo test --features "visualize,io" # feature-gated code paths

# Single test
cargo test test_simulate_bm                      # by name substring
cargo test simulation::continuous::bm::tests     # by module path
cargo test test_simulate_bm -- --exact --nocapture

cargo run --example bm               # examples: bm, langevin, randoms, CIR
cargo bench --bench bm               # benches: bm, ou, random, levy (Criterion)
```

GPU builds require a toolchain and are feature-gated: `cargo build --features cuda`
(needs `nvcc`) or `cargo build --features metal` (needs Xcode). `cuda` and `metal` are
mutually exclusive — `build.rs` panics if both are enabled.

Pre-commit (`.pre-commit-config.yaml`) runs `cargo fmt --check`, `cargo check --all`, and
the clippy command above. CI (`.github/workflows/test.yml`) runs build + test only.

## Architecture

DiffusionX is a Rust library (edition 2024, MSRV 1.88) for random number generation and
stochastic process simulation. The design is trait-based: implement one base trait + a `simulate` method and
a process gets the full suite of observables (moments, first passage time, occupation
time, TAMSD, plotting) for free via default trait methods.

### Trait + trajectory layering

- **`src/simulation/basic/`** — the core abstractions: base traits `ContinuousProcess`,
  `PointProcess`, `DiscreteProcess`; the `Moment` trait; `tamsd`, `inverse`, `functional`
  helpers. Concrete process structs live in **`continuous/`**, **`discrete/`**, **`point/`**
  and implement these traits.
- A bare process (e.g. `Bm`) only knows how to `simulate` a single path. Ensemble
  statistics work through a **trajectory**: `process.duration(t)` returns a
  `ContinuousTrajectory` (likewise `DiscreteTrajectory`, `PointTrajectory`). The `Moment`
  trait is implemented on the *trajectory* types, not the processes — so `mean`, `msd`,
  `raw_moment`, fractional moments, etc. are reached via `.duration(t)?` first.
- `ContinuousProcess` also exposes convenience methods (`mean`, `msd`, `fpt`, `eatamsd`…)
  that internally build a trajectory; these carry a `where Self: ContinuousTrajectoryTrait`
  bound.

### Numeric generics

`src/lib.rs` defines three blanket-implemented trait aliases — `FloatExt` (float-only
processes), `IntExt` (discrete integer states), `RealExt` (int-or-float state values).
Use these bounds rather than re-listing `num_traits` requirements. Continuous processes
are generic over `T: FloatExt` and default to `f64`.

### Errors

`src/error.rs` defines `XError` (the crate-wide error enum), `StableError`,
`SimulationError`, the `XResult<T>` alias, and validation helpers like
`check_duration_time_step`. Public fallible APIs return `XResult<T>`.

### Feature-gated modules

- `visualize` — `src/visualize/`, plotting via `plotters`; adds `Visualize`/`plot`.
- `io` — CSV helpers in `src/utils/csv.rs`.
- `cuda` / `metal` — `src/gpu/{cuda,metal}/`. `build.rs` compiles the kernel sources in
  `kernels/cuda-kernel/*.cu` (→ PTX via `nvcc`) or `kernels/metal-kernel/*.metal`
  (→ metallib via `xcrun`) and passes the artifact paths to the crate as env vars
  (e.g. `BM_KERNEL_PTX`, `BM_KERNEL_METALLIB`). The `GPUMoment` trait operates on `f32`
  and is implemented only for `Bm`, `OrnsteinUhlenbeck`, and `Levy`.
- `mimalloc` (default) — sets the global allocator.

### Other notes

- RNG: all distributions share a Xoshiro256++ entropy source. Modules in `src/random/`
  expose free functions (`rand`, `rands`, `standard_rands`, …).
- `langevin!`, `generalized_langevin!`, `subordinated_langevin!`
  (`src/simulation/macros.rs`) are `#[macro_export]`ed DSL-style constructors available
  at the crate root.
- Parallelism uses `rayon`; loops switch between sequential and parallel based on a
  `PAR_THRESHOLD`. Keep that pattern when adding ensemble computations.
- The crate's rustdoc landing page is `README.md` (included via `#![doc = include_str!]`),
  so README changes affect generated docs.

To add a new continuous process, implement `ContinuousProcess` (define `start` and
`simulate`); the moment/FPT/TAMSD/plot machinery comes from default methods. See
`examples/CIR.rs` for a worked example.
