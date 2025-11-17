//! CUDA kernel PTX bindings for stochastic process simulation
//!
//! This crate exposes compiled PTX code for various stochastic processes.

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

// Export PTX strings for each kernel module
pub const BM_PTX: &str = ptx::BM;
pub const OU_PTX: &str = ptx::OU;
pub const GBM_PTX: &str = ptx::GBM;
pub const STATS_PTX: &str = ptx::STATS;
pub const FBM_PTX: &str = ptx::FBM;
pub const CAUCHY_PTX: &str = ptx::CAUCHY;
pub const GAMMA_PTX: &str = ptx::GAMMA;
pub const LANGEVIN_PTX: &str = ptx::LANGEVIN;
pub const LEVY_PTX: &str = ptx::LEVY;
pub const LEVY_WALK_PTX: &str = ptx::LEVY_WALK;
pub const BROWNIAN_BRIDGE_PTX: &str = ptx::BROWNIAN_BRIDGE;
pub const BROWNIAN_EXCURSION_PTX: &str = ptx::BROWNIAN_EXCURSION;
pub const BROWNIAN_MEANDER_PTX: &str = ptx::BROWNIAN_MEANDER;
pub const BNG_PTX: &str = ptx::BNG;
