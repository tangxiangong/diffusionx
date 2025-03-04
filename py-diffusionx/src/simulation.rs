use crate::XPyResult;
use diffusionx::simulation::{
    Bm, CTRW, Fbm, InvSubordinator, Levy, Poisson, Subordinator, prelude::*,
};
use numpy::{IntoPyArray, Ix1, PyArray};
use pyo3::prelude::*;

type PyArrayPair<'py> = (Bound<'py, PyArray<f64, Ix1>>, Bound<'py, PyArray<f64, Ix1>>);

type PyArrayPointPair<'py> = (Bound<'py, PyArray<f64, Ix1>>, Bound<'py, PyArray<i64, Ix1>>);

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

#[pyfunction]
pub fn bm_occupation_time(
    start_position: f64,
    diffusion_coefficient: f64,
    step_size: f64,
    domain: (f64, f64),
    duration: f64,
) -> XPyResult<f64> {
    let bm = Bm::new(start_position, diffusion_coefficient)?;
    let result = bm.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn levy_simulate(
    py: Python,
    start_position: f64,
    alpha: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let levy = Levy::new(start_position, alpha)?;
    let (times, positions) = levy.simulate(duration, step_size)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn levy_fpt(
    start_position: f64,
    alpha: f64,
    step_size: f64,
    domain: (f64, f64),
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let levy = Levy::new(start_position, alpha)?;
    let result = levy.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn levy_occupation_time(
    start_position: f64,
    alpha: f64,
    step_size: f64,
    domain: (f64, f64),
    duration: f64,
) -> XPyResult<f64> {
    let levy = Levy::new(start_position, alpha)?;
    let result = levy.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn poisson_simulate_duration(
    py: Python,
    lambda: f64,
    duration: f64,
) -> XPyResult<PyArrayPointPair<'_>> {
    let poisson = Poisson::new(lambda)?;
    let (times, positions) = poisson.simulate_with_duration(duration)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn poisson_simulate_step(
    py: Python,
    lambda: f64,
    num_step: usize,
) -> XPyResult<PyArrayPointPair<'_>> {
    let poisson = Poisson::new(lambda)?;
    let (times, positions) = poisson.simulate_with_step(num_step)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn poisson_raw_moment(
    lambda: f64,
    duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let poisson = Poisson::new(lambda)?;
    let result = poisson.raw_moment(duration, order, particles)?;
    Ok(result)
}

#[pyfunction]
pub fn poisson_central_moment(
    lambda: f64,
    duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let poisson = Poisson::new(lambda)?;
    let result = poisson.central_moment(duration, order, particles)?;
    Ok(result)
}

#[pyfunction]
pub fn poisson_fpt(lambda: f64, domain: (f64, f64), max_duration: f64) -> XPyResult<Option<f64>> {
    let poisson = Poisson::new(lambda)?;
    let result = poisson.fpt(domain, max_duration)?;
    Ok(result)
}

#[pyfunction]
pub fn poisson_occupation_time(lambda: f64, domain: (f64, f64), duration: f64) -> XPyResult<f64> {
    let poisson = Poisson::new(lambda)?;
    let result = poisson.occupation_time(domain, duration)?;
    Ok(result)
}

#[pyfunction]
pub fn subordinator_simulate(
    py: Python,
    alpha: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let subordinator = Subordinator::new(alpha)?;
    let (times, positions) = subordinator.simulate(duration, step_size)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn subordinator_fpt(
    alpha: f64,
    domain: (f64, f64),
    max_duration: f64,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let subordinator = Subordinator::new(alpha)?;
    let result = subordinator.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn subordinator_occupation_time(
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    step_size: f64,
) -> XPyResult<f64> {
    let subordinator = Subordinator::new(alpha)?;
    let result = subordinator.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn inv_subordinator_simulate(
    py: Python,
    alpha: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let inv_subordinator = InvSubordinator::new(alpha)?;
    let (times, positions) = inv_subordinator.simulate(duration, step_size)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn inv_subordinator_fpt(
    alpha: f64,
    domain: (f64, f64),
    max_duration: f64,
    step_size: f64,
) -> XPyResult<Option<f64>> {
    let inv_subordinator = InvSubordinator::new(alpha)?;
    let result = inv_subordinator.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn inv_subordinator_occupation_time(
    alpha: f64,
    domain: (f64, f64),
    duration: f64,
    step_size: f64,
) -> XPyResult<f64> {
    let inv_subordinator = InvSubordinator::new(alpha)?;
    let result = inv_subordinator.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn fbm_simulate(
    py: Python,
    start_position: f64,
    hurst_exponent: f64,
    duration: f64,
    step_size: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let fbm = Fbm::new(start_position, hurst_exponent)?;
    let (times, positions) = fbm.simulate(duration, step_size)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn fbm_raw_moment(
    start_position: f64,
    hurst_exponent: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let fbm = Fbm::new(start_position, hurst_exponent)?;
    let result = fbm.raw_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn fbm_central_moment(
    start_position: f64,
    hurst_exponent: f64,
    duration: f64,
    step_size: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let fbm = Fbm::new(start_position, hurst_exponent)?;
    let result = fbm.central_moment(duration, order, particles, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn fbm_fpt(
    start_position: f64,
    hurst_exponent: f64,
    step_size: f64,
    domain: (f64, f64),
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let fbm = Fbm::new(start_position, hurst_exponent)?;
    let result = fbm.fpt(domain, max_duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn fbm_occupation_time(
    start_position: f64,
    hurst_exponent: f64,
    step_size: f64,
    domain: (f64, f64),
    duration: f64,
) -> XPyResult<f64> {
    let fbm = Fbm::new(start_position, hurst_exponent)?;
    let result = fbm.occupation_time(domain, duration, step_size)?;
    Ok(result)
}

#[pyfunction]
pub fn ctrw_simulate_duration(
    py: Python,
    alpha: f64,
    beta: f64,
    start_position: f64,
    duration: f64,
) -> XPyResult<PyArrayPair<'_>> {
    let ctrw = CTRW::new(alpha, beta, start_position)?;
    let (times, positions) = ctrw.simulate(duration)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn ctrw_simulate_step(
    py: Python,
    alpha: f64,
    beta: f64,
    start_position: f64,
    num_step: usize,
) -> XPyResult<PyArrayPair<'_>> {
    let ctrw = CTRW::new(alpha, beta, start_position)?;
    let (times, positions) = ctrw.simulate_with_step(num_step)?;
    let times_array = times.into_pyarray(py);
    let positions_array = positions.into_pyarray(py);
    Ok((times_array, positions_array))
}

#[pyfunction]
pub fn ctrw_raw_moment(
    alpha: f64,
    beta: f64,
    start_position: f64,
    duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let ctrw = CTRW::new(alpha, beta, start_position)?;
    let result = ctrw.raw_moment(duration, order, particles)?;
    Ok(result)
}

#[pyfunction]
pub fn ctrw_central_moment(
    alpha: f64,
    beta: f64,
    start_position: f64,
    duration: f64,
    order: i32,
    particles: usize,
) -> XPyResult<f64> {
    let ctrw = CTRW::new(alpha, beta, start_position)?;
    let result = ctrw.central_moment(duration, order, particles)?;
    Ok(result)
}

#[pyfunction]
pub fn ctrw_fpt(
    alpha: f64,
    beta: f64,
    start_position: f64,
    domain: (f64, f64),
    max_duration: f64,
) -> XPyResult<Option<f64>> {
    let ctrw = CTRW::new(alpha, beta, start_position)?;
    let result = ctrw.fpt(domain, max_duration)?;
    Ok(result)
}

#[pyfunction]
pub fn ctrw_occupation_time(
    alpha: f64,
    beta: f64,
    start_position: f64,
    domain: (f64, f64),
    duration: f64,
) -> XPyResult<f64> {
    let ctrw = CTRW::new(alpha, beta, start_position)?;
    let result = ctrw.occupation_time(domain, duration)?;
    Ok(result)
}
