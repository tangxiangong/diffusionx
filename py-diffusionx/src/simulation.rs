use crate::XPyResult;
use diffusionx::simulation::*;
use numpy::{IntoPyArray, Ix1, PyArray};
use pyo3::prelude::*;

type PyArrayPair<'py> = (Bound<'py, PyArray<f64, Ix1>>, Bound<'py, PyArray<f64, Ix1>>);

#[pyfunction]
pub fn bm_simulate(
    py: Python,
    start_position: f64,
    diffusion_coefficient: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let (times, positions) = bm.simulate(duration, step_size)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn bm_raw_moment(
    start_position: f64,
    diffusion_coefficient: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let result = bm.raw_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn bm_central_moment(
    start_position: f64,
    diffusion_coefficient: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let result = bm.central_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn bm_fpt(
    start_position: f64,
    diffusion_coefficient: f64,
    step_size: f64,
    domain: (f64, f64),
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let result = bm.fpt(domain, max_duration, step_size)?;
    Ok(result)
}
