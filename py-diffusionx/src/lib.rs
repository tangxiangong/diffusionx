use pyo3::prelude::*;

pub mod random;
mod error;
pub use error::*;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .unwrap();
    m.add_function(wrap_pyfunction!(random::exp_rand, m)?)?;
    m.add_function(wrap_pyfunction!(random::exp_rands, m)?)?;
    m.add_function(wrap_pyfunction!(random::uniform_rand_float, m)?)?;
    m.add_function(wrap_pyfunction!(random::uniform_rand_int, m)?)?;
    m.add_function(wrap_pyfunction!(random::uniform_rands_float, m)?)?;
    m.add_function(wrap_pyfunction!(random::uniform_rands_int, m)?)?;
    m.add_function(wrap_pyfunction!(random::normal_rand, m)?)?;
    m.add_function(wrap_pyfunction!(random::normal_rands, m)?)?;
    m.add_function(wrap_pyfunction!(random::poisson_rand, m)?)?;
    m.add_function(wrap_pyfunction!(random::poisson_rands, m)?)?;
    Ok(())
}
