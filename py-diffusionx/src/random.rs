use crate::XPyResult;
use diffusionx::{
    XResult,
    random::{exponential, normal, poisson, stable, uniform},
};
use numpy::{IntoPyArray, Ix1, PyArray};
use pyo3::prelude::*;
use rand::distr::uniform::{SampleUniform, Uniform};

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
#[pyo3(signature = (low = 0.0, high = 1.0, /, end = false))]
pub fn uniform_rand_float(low: f64, high: f64, end: bool) -> XPyResult<f64> {
    let result = if low == 0.0 && high == 1.0 {
        _uniform_rand_with_end(0.0, 1.0, end)?
    } else {
        _uniform_rand_with_end(low, high, end)?
    };
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (low, high, /, end = false))]
pub fn uniform_rand_int(low: i64, high: i64, end: bool) -> XPyResult<i64> {
    let result = _uniform_rand_with_end(low, high, end)?;
    Ok(result)
}

fn _uniform_rand_with_end<T>(low: T, high: T, end: bool) -> XResult<T>
where
    T: SampleUniform + Send + Sync,
    Uniform<T>: Copy,
    <T as SampleUniform>::Sampler: Send + Sync,
{
    if end {
        Ok(uniform::inclusive_range_rand(low..=high)?)
    } else {
        Ok(uniform::range_rand(low..high)?)
    }
}

#[pyfunction]
#[pyo3(signature = (n, /, low = 0.0, high = 1.0, end = false))]
pub fn uniform_rands_float(
    py: Python,
    n: usize,
    low: f64,
    high: f64,
    end: bool,
) -> XPyResult<Bound<'_, PyArray<f64, Ix1>>> {
    let result = if low == 0.0 && high == 1.0 {
        _uniform_rands_with_end(n, 0.0, 1.0, end)?
    } else {
        _uniform_rands_with_end(n, low, high, end)?
    };
    let result = result.into_pyarray(py);
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (n, low, high, /, end = false))]
pub fn uniform_rands_int(
    py: Python,
    n: usize,
    low: i64,
    high: i64,
    end: bool,
) -> XPyResult<Bound<'_, PyArray<i64, Ix1>>> {
    let result = _uniform_rands_with_end(n, low, high, end)?;
    let result = result.into_pyarray(py);
    Ok(result)
}

fn _uniform_rands_with_end<T>(n: usize, low: T, high: T, end: bool) -> XResult<Vec<T>>
where
    T: SampleUniform + Send + Sync,
    Uniform<T>: Copy,
    <T as SampleUniform>::Sampler: Send + Sync,
{
    if end {
        Ok(uniform::inclusive_range_rands(low..=high, n)?)
    } else {
        Ok(uniform::range_rands(low..high, n)?)
    }
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
pub fn normal_rands(
    py: Python,
    n: usize,
    mu: f64,
    sigma: f64,
) -> XPyResult<Bound<'_, PyArray<f64, Ix1>>> {
    let result = if mu == 0.0 && sigma == 1.0 {
        normal::standard_rands(n)
    } else {
        normal::rands(mu, sigma, n)?
    };
    let result = result.into_pyarray(py);
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (lambda_ = 1.0))]
pub fn poisson_rand(lambda_: f64) -> XPyResult<u64> {
    let result = poisson::rand(lambda_)?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (n, /, lambda_ = 1.0))]
pub fn poisson_rands(
    py: Python,
    n: usize,
    lambda_: f64,
) -> XPyResult<Bound<'_, PyArray<u64, Ix1>>> {
    let result = poisson::rands(lambda_, n)?;
    let result = result.into_pyarray(py);
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (alpha, beta, /, sigma = 1.0, mu = 0.0))]
pub fn stable_rand(alpha: f64, beta: f64, sigma: f64, mu: f64) -> XPyResult<f64> {
    let result = if sigma == 1.0 && mu == 0.0 {
        stable::standard_rand(alpha, beta)?
    } else {
        stable::rand(alpha, beta, sigma, mu)?
    };
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (n, /, alpha, beta, sigma = 1.0, mu = 0.0))]
pub fn stable_rands(
    py: Python,
    n: usize,
    alpha: f64,
    beta: f64,
    sigma: f64,
    mu: f64,
) -> XPyResult<Bound<'_, PyArray<f64, Ix1>>> {
    let result = stable::rands(alpha, beta, sigma, mu, n)?;
    let result = result.into_pyarray(py);
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (alpha))]
pub fn skew_stable_rand(alpha: f64) -> XPyResult<f64> {
    let result = stable::skew_rand(alpha)?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (n, alpha))]
pub fn skew_stable_rands(
    py: Python,
    n: usize,
    alpha: f64,
) -> XPyResult<Bound<'_, PyArray<f64, Ix1>>> {
    let result = stable::skew_rands(alpha, n)?;
    let result = result.into_pyarray(py);
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (p = 0.5))]
pub fn bool_rand(p: f64) -> XPyResult<bool> {
    let result = uniform::bool_rand(p)?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (n, /, p = 0.5))]
pub fn bool_rands(py: Python, n: usize, p: f64) -> XPyResult<Bound<'_, PyArray<bool, Ix1>>> {
    let result = uniform::bool_rands(p, n)?;
    let result = result.into_pyarray(py);
    Ok(result)
}
