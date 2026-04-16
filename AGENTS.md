# AGENTS.md

Guidance for coding agents working in `diffusionx`.

## Rule Sources (Cursor/Copilot)

- Checked `.cursorrules`: not present.
- Checked `.cursor/rules/`: not present.
- Checked `.github/copilot-instructions.md`: not present.
- No additional Cursor or Copilot instruction files are currently defined.

## Project Facts

- Language: Rust (edition `2024`).
- Crate: `diffusionx`.
- MSRV: `1.87` (`rust-version` in `Cargo.toml`).
- Library doctests: disabled (`[lib] doctest = false`).
- Default feature: `mimalloc`.
- Optional features: `cuda`, `metal`, `io`, `visualize`.
- `cuda` and `metal` are mutually exclusive (enforced by build logic).

## Repository Map

- `src/lib.rs`: crate root, trait aliases (`FloatExt`, `IntExt`, `RealExt`), module exports.
- `src/error.rs`: `XError`, `SimulationError`, `StableError`, `XResult<T>`, validation helpers.
- `src/random/`: distribution modules and RNG helpers.
- `src/simulation/basic/`: core traits (`ContinuousProcess`, etc.) and shared abstractions.
- `src/simulation/continuous/`, `src/simulation/discrete/`, `src/simulation/point/`: process implementations.
- `src/gpu/`: backend-specific GPU code (`cuda` / `metal` gated).
- `src/visualize/`: plotting support (feature-gated).
- `src/utils/`: utility functions and optional CSV helpers.
- `kernels/`: CUDA/Metal kernel sources used by `build.rs`.
- `examples/` and `benches/`: runnable examples and Criterion benchmarks.

## Build, Lint, and Test Commands

Run the narrowest command that validates your change first.

### Build

- `cargo build`
- `cargo build --release`
- `cargo check --all`
- `cargo build --features "visualize,io"`
- `cargo build --no-default-features`
- `cargo build --features metal` (requires Apple Metal toolchain)
- `cargo build --features cuda` (requires CUDA/nvcc)

### Format and Lint

- `cargo fmt -- --check`
- `cargo fmt`
- `cargo clippy --all-targets --tests --benches -- -D warnings`
- `cargo clippy --all-targets --features "visualize,io" --tests --benches -- -D warnings`
- `cargo clippy --all-targets --features metal --tests --benches -- -D warnings` (requires Apple Metal toolchain)
- `cargo clippy --all-targets --features cuda --tests --benches -- -D warnings` (requires CUDA/nvcc)

Pre-commit mirrors these checks (`.pre-commit-config.yaml`) with hooks for:

- `cargo fmt -- --check`
- `cargo check --all`
- `cargo clippy --all-targets --tests --benches -- -D warnings`

### Tests

- `cargo test --verbose`
- `cargo test`
- `cargo test --features "visualize,io"`
- `cargo test --features metal` (requires Apple Metal toolchain)
- `cargo test --features cuda` (requires CUDA/nvcc)
- `cargo test --lib`
- `cargo test --features "visualize,io"`

### Run a Single Test (important)

- By test name substring: `cargo test test_standard_rand`
- By fully qualified path: `cargo test random::normal::tests::test_standard_rand`
- Restrict to library tests while filtering: `cargo test --lib test_standard_rand`
- Exact name match only: `cargo test test_standard_rand -- --exact`
- Show captured output: `cargo test test_standard_rand -- --nocapture`
- Single-threaded execution (debugging flaky behavior): `cargo test test_standard_rand -- --test-threads=1`

### Examples and Benches

- Run one example: `cargo run --example bm`
- Build all examples: `cargo build --examples`
- Run benches: `cargo bench` or `cargo bench --bench bm`

### Docs

- `cargo doc --no-deps`
- `cargo doc --features "visualize,io" --no-deps`

## Style and Design Conventions

### Formatting and Structure

- Use rustfmt defaults; do not hand-format against formatter output.
- Keep modules cohesive; prefer extending nearby code over introducing parallel abstractions.
- Favor small focused functions for simulation/math operations.
- Keep public APIs stable unless change is explicitly requested.

### Imports

- Group imports in existing style (crate imports first, then external crates).
- Prefer explicit imports used in file (`use crate::{...}`) over wildcard for crate internals.
- `rayon::prelude::*` is acceptable where parallel iterators are used.
- Avoid adding unused imports; keep clippy/rustc warnings clean.

### Types and Generics

- Reuse crate trait bounds (`FloatExt`, `IntExt`, `RealExt`) instead of ad-hoc bounds.
- For continuous processes, generic `T: FloatExt` is the common pattern.
- Keep structs `Debug + Clone` when consistent with neighboring process types.
- Preserve `Send + Sync` compatibility when practical.

### Naming

- Types/traits: `UpperCamelCase` (`Bm`, `ContinuousProcess`).
- Functions/methods/modules: `snake_case`.
- Keep naming aligned with existing domain terms: `simulate`, `displacement`, `msd`, `fpt`, `tamsd`.
- Constructor/getter patterns should match existing style (`new`, `get_*`, `default`).

### Error Handling

- Public fallible APIs should return `XResult<T>`.
- Prefer existing error enums/variants in `src/error.rs`.
- Validate parameters early and return structured errors.
- Use helper validators (for example `check_duration_time_step`) when applicable.
- Avoid `unwrap()` in public/library paths unless invariant is guaranteed internally.

### Parallelism and Performance

- Follow existing sequential vs parallel thresholds (`PAR_THRESHOLD`).
- Use `rayon` for data-parallel loops in the established style (`into_par_iter`, `map_init`).
- Avoid unnecessary allocations/clones in hot loops.
- Keep numeric algorithms robust; avoid brittle exact-equality checks in floating-point tests.

### Features and Conditional Compilation

- Gate optional functionality explicitly with `#[cfg(feature = "...")]`.
- Keep `cuda` and `metal` paths mirrored when behavior is meant to be equivalent.
- Do not introduce unconditional dependencies on optional crates.
- For docs-facing feature APIs, follow existing `cfg_attr(docsrs, doc(cfg(...)))` style where used.

### Documentation and Examples

- Add concise rustdoc for public items and changed behaviors.
- If public API changes, update `README.md` and/or examples accordingly.
- Remember crate docs include `README.md`; README drift affects generated docs.

### Testing Expectations

- Add tests next to modified code (`#[cfg(test)] mod tests`).
- Prefer behavior/invariant-focused tests and tolerances for numeric results.
- Keep tests deterministic where possible; if randomness is involved, assert statistical sanity.
- Use the smallest test scope first, then broaden (`--lib`, full suite, feature builds).

## CI Reference

- GitHub Actions (`.github/workflows/test.yml`) currently runs:
  - `cargo build --verbose`
  - `cargo test --verbose`
- CI also installs plotting-related system libs on Ubuntu (`libfreetype6-dev`, `libfontconfig1-dev`).
