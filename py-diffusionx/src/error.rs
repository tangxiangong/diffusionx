use diffusionx::XError;
use pyo3::{PyErr, exceptions::PyValueError};
use thiserror::Error;

pub type XPyResult<T> = Result<T, XPyError>;

#[derive(Error, Debug, PartialEq, Eq, Ord, PartialOrd)]
pub enum XPyError {
    #[error("Invalid value: {0}")]
    ValueError(String),
}

impl From<XError> for XPyError {
    fn from(error: XError) -> Self {
        XPyError::ValueError(error.to_string())
    }
}

impl From<XPyError> for PyErr {
    fn from(error: XPyError) -> Self {
        PyValueError::new_err(error.to_string())
    }
}
