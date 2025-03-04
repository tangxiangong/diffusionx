use pyo3::prelude::*;

mod error;
pub mod random;
pub use error::*;
pub mod simulation;

/// 用于简化 PyModule 函数注册的宏
macro_rules! register_functions {
    // 匹配 a::b::c 形式
    ($m:ident, $($p1:ident::$p2:ident::$func:ident),* $(,)?) => {
        $(
            $m.add_function(wrap_pyfunction!($p1::$p2::$func, $m)?)?;
        )*
    };
    // 匹配 a::b 形式
    ($m:ident, $($p1:ident::$func:ident),* $(,)?) => {
        $(
            $m.add_function(wrap_pyfunction!($p1::$func, $m)?)?;
        )*
    };
    // 匹配单独的 a 形式
    ($m:ident, $($func:ident),* $(,)?) => {
        $(
            $m.add_function(wrap_pyfunction!($func, $m)?)?;
        )*
    };
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    register_functions!(
        m,
        random::exp_rand,
        random::exp_rands,
        random::uniform_rand_float,
        random::uniform_rand_int,
        random::uniform_rands_float,
        random::uniform_rands_int,
        random::normal_rand,
        random::normal_rands,
        random::poisson_rand,
        random::poisson_rands,
        random::stable_rand,
        random::stable_rands,
        random::skew_stable_rand,
        random::skew_stable_rands,
        random::bool_rand,
        random::bool_rands,
        simulation::bm_simulate,
        simulation::bm_raw_moment,
        simulation::bm_central_moment,
        simulation::bm_fpt,
        simulation::bm_occupation_time,
        simulation::fbm_simulate,
        simulation::fbm_raw_moment,
        simulation::fbm_central_moment,
        simulation::fbm_fpt,
        simulation::fbm_occupation_time,
        simulation::ctrw_simulate_duration,
        simulation::ctrw_simulate_step,
        simulation::ctrw_raw_moment,
        simulation::ctrw_central_moment,
        simulation::ctrw_fpt,
        simulation::ctrw_occupation_time,
    );

    Ok(())
}
