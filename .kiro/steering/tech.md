# Tech Stack

## Language & Edition
- Rust 2024 edition, MSRV 1.88
- GPU kernels: CUDA (.cu) and Metal (.metal)

## Build System
- Cargo (standard Rust toolchain)
- Custom `build.rs` compiles GPU kernels:
  - CUDA: uses `nvcc` to produce PTX files
  - Metal: uses `xcrun metal` / `xcrun metallib` to produce .metallib files
- CUDA and Metal features are mutually exclusive (build.rs panics if both enabled)

## Key Dependencies
- `rayon` — parallel iteration for Monte Carlo simulations
- `rand`, `rand_distr`, `rand_xoshiro` — RNG (Xoshiro256++ as the common entropy source)
- `num-traits`, `num-complex` — generic numeric traits
- `realfft` — FFT for fractional Brownian motion (circulant embedding)
- `gauss-quad` — Gauss-Legendre quadrature for TAMSD
- `thiserror` — error type derivation
- `mimalloc` — default global allocator (feature-gated, on by default)
- `cudarc` — CUDA runtime bindings (optional)
- `metal` — Metal API bindings (optional)
- `plotters` / `derive_builder` — visualization (optional)
- `csv` — CSV I/O (optional)
- `criterion` — benchmarks (dev-dependency)

## Cargo Features
| Feature     | Purpose                          | Default |
| ----------- | -------------------------------- | ------- |
| `mimalloc`  | Use mimalloc as global allocator | Yes     |
| `cuda`      | CUDA GPU acceleration            | No      |
| `metal`     | Metal GPU acceleration (macOS)   | No      |
| `io`        | CSV read/write support           | No      |
| `visualize` | Trajectory plotting via plotters | No      |

## Common Commands
```bash
# Build (default features)
cargo build

# Build with all non-GPU optional features
cargo build --features "visualize,io"

# Build with Metal GPU support (macOS only)
cargo build --features metal

# Run tests
cargo test

# Run benchmarks
cargo bench

# Run a specific example
cargo run --example bm

# Generate docs (with feature flags matching docs.rs config)
cargo doc --features "visualize,io" --no-deps

# Check formatting
cargo fmt --check

# Lint
cargo clippy
```

## Release Profile
LTO enabled, single codegen unit, panic=abort.
