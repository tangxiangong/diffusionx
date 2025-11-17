//! Metal kernel source code for stochastic process simulation
//!
//! This crate exposes Metal shader source code as compile-time strings.

// Export Metal shader source code
pub const BM_METAL: &str = include_str!("metal_src/bm.metal");
pub const OU_METAL: &str = include_str!("metal_src/ou.metal");
pub const GBM_METAL: &str = include_str!("metal_src/geometric_bm.metal");
pub const STATS_METAL: &str = include_str!("metal_src/stats.metal");
pub const FBM_METAL: &str = include_str!("metal_src/fbm.metal");
pub const CAUCHY_METAL: &str = include_str!("metal_src/cauchy.metal");
pub const GAMMA_METAL: &str = include_str!("metal_src/gamma.metal");
pub const LANGEVIN_METAL: &str = include_str!("metal_src/langevin.metal");
pub const LEVY_METAL: &str = include_str!("metal_src/levy.metal");
pub const LEVY_WALK_METAL: &str = include_str!("metal_src/levy_walk.metal");
pub const BROWNIAN_BRIDGE_METAL: &str = include_str!("metal_src/brownian_bridge.metal");
pub const BROWNIAN_EXCURSION_METAL: &str = include_str!("metal_src/brownian_excursion.metal");
pub const BROWNIAN_MEANDER_METAL: &str = include_str!("metal_src/brownian_meander.metal");
pub const BNG_METAL: &str = include_str!("metal_src/bng.metal");
pub const STABLE_METAL: &str = include_str!("metal_src/stable.metal");

/// Get all Metal shader sources combined
pub fn all_shaders() -> String {
    vec![
        BM_METAL,
        OU_METAL,
        GBM_METAL,
        STATS_METAL,
        FBM_METAL,
        CAUCHY_METAL,
        GAMMA_METAL,
        LANGEVIN_METAL,
        LEVY_METAL,
        LEVY_WALK_METAL,
        BROWNIAN_BRIDGE_METAL,
        BROWNIAN_EXCURSION_METAL,
        BROWNIAN_MEANDER_METAL,
        BNG_METAL,
        STABLE_METAL,
    ]
    .join("\n\n")
}
