# DiffusionX

DiffusionX is a high-performance Rust library (`diffusionx` crate) for random number generation and stochastic process simulation, with optional GPU acceleration via CUDA and Metal.

## Core Capabilities
- Random number generation from various distributions (Normal, Uniform, Exponential, Poisson, α-stable, Gamma)
- Simulation of continuous, discrete, and point stochastic processes (Brownian motion, Lévy processes, Ornstein-Uhlenbeck, Langevin equations, fractional Brownian motion, etc.)
- Statistical moment computation (raw, central, fractional, MSD, TAMSD, EATAMSD)
- First passage time and occupation time calculations
- GPU-accelerated moment calculations (CUDA on Linux, Metal on macOS)
- Trajectory visualization (optional `visualize` feature)
- CSV I/O (optional `io` feature)

## Target Audience
Researchers and developers working in stochastic calculus, statistical physics, and Monte Carlo simulation.

## Licensing
Dual-licensed under MIT and Apache-2.0.
