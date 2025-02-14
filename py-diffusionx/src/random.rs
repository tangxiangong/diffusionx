use pyo3::prelude::*;
use diffusionx::random::{exponential, uniform, normal};
use crate::XPyResult;
use numpy::{PyArray, IntoPyArray, Ix1};

#[pyfunction]
#[pyo3(signature = (scale = 1.0))]
pub fn exp_rand(scale: f64) -> XPyResult<f64> {
    let result = if scale == 1.0 {
        exponential::standard_rand()
    } else {
        exponential::rand(1.0 / scale)?
    };
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (n, /, scale = 1.0))]
pub fn exp_rands(py: Python, n: usize, scale: f64) -> XPyResult<Bound<'_, PyArray<f64, Ix1>>> {
    let result = if scale == 1.0 {
        exponential::standard_rands(n)
    } else {
        exponential::rands(1.0 / scale, n)?
    };
    let result = result.into_pyarray(py);
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (low = 0.0, high = 1.0))]
pub fn uniform_rand(low: f64, high: f64) -> XPyResult<f64> {
    let result =  if low == 0.0 && high == 1.0 {
        uniform::standard_rand()
    } else {
        uniform::range_rand(low..high)?
    };
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (n, /, low = 0.0, high = 1.0))]
pub fn uniform_rands(py: Python, n: usize, low: f64, high: f64) -> XPyResult<Bound<'_, PyArray<f64, Ix1>>> {
    let result =  if low == 0.0 && high == 1.0 {
        uniform::standard_rands(n)
    } else {
        uniform::range_rands(low..high, n)?
    };
    let result = result.into_pyarray(py);
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (mu = 0.0, sigma = 1.0))]
pub fn normal_rand(mu: f64, sigma: f64) -> XPyResult<f64> {
    let result = if mu == 0.0 && sigma == 1.0 {
        normal::standard_rand()
    } else {
        normal::rand(mu, sigma)?
    };
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (n, /, mu = 0.0, sigma = 1.0))]
pub fn normal_rands(py: Python, n: usize, mu: f64, sigma: f64) -> XPyResult<Bound<'_, PyArray<f64, Ix1>>> {
    let result =  if mu == 0.0 && sigma == 1.0 {
        normal::standard_rands(n)
    } else {
        normal::rands(mu, sigma, n)?
    };
    let result = result.into_pyarray(py);
    Ok(result)
}


