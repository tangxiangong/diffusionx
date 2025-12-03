//! CUDA kernel PTX bindings for stochastic process simulation
//!
//! This crate exposes compiled PTX code for various stochastic processes.

pub const BM_PTX: &str = include_str!(env!("BM_KERNEL_PTX"));
pub const STABLE_PTX: &str = include_str!(env!("STABLE_KERNEL_PTX"));
