//! # Auxiliary functions
//!
//! This module provides auxiliary functions.
//!
//! ## Functions
//!
//! - `cumsum`: Calculate the cumulative sum of a vector.
//! - `approx_eq`: Check if two numbers are approximately equal.
//! - `float_eq`: Check if two numbers are equal.
//! - `eval_poly`: Evaluate a polynomial.
//! - `minmax`: Find the minimum and maximum values in a vector.
//! - `calculate_stats`: Calculate the mean and variance of an array.
//! - `calculate_int_stats`: Calculate the mean and variance of an integer array.
//! - `calculate_bool_mean`: Calculate the mean of a boolean array.
//!

use crate::{PlotterError, XResult};
use num_traits::Num;
use std::path::Path;

/// Calculate the cumulative sum of a vector
///
/// Returns a vector of cumulative sums
///
/// # Arguments
///
/// * `start` - The initial value of the cumulative sum
/// * `v` - The vector to calculate the cumulative sum of
///
/// # Example
///
/// ```rust
/// use diffusionx::utils::cumsum;
///
/// let v = vec![1, 2, 3, 4, 5];
/// let result = cumsum(0, &v);
/// assert_eq!(result, vec![0, 1, 3, 6, 10, 15]);
/// ```
pub fn cumsum<T>(start: T, v: &[T]) -> Vec<T>
where
    T: Num + Copy,
{
    if v.is_empty() {
        return Vec::new();
    }
    std::iter::once(start)
        .chain(v.iter().scan(start, |acc, x| {
            *acc = *acc + *x;
            Some(*acc)
        }))
        .collect()
}

/// Check if two floating numbers are approximately equal within a tolerance
///
/// # Arguments
///
/// * `a` - The first number
/// * `b` - The second number
/// * `tol` - The tolerance
///
/// # Example
///
/// ```rust
/// use diffusionx::utils::approx_eq;
///
/// let a = 1.0;
/// let b = 1.0;
/// let result = approx_eq(a, b, 1.0e-6);
/// assert!(result);
/// ```
#[inline]
pub fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    let result = if a.is_infinite() || b.is_infinite() {
        false
    } else {
        (a - b).abs() <= tol
    };
    if !result {
        println!("The left is {}", a);
        println!("The right is {}", b);
        println!(
            "These two numbers are not approximately equal with tol {}",
            tol
        );
    }
    result
}

/// Ensure the output directory exists, or create it if it doesn't exist.
pub(crate) fn ensure_output_dir(path: &Path) -> XResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| PlotterError::ConfigError(e.to_string()))?;
    }
    Ok(())
}

/// Check if two floating numbers are equal within the f64 precision
///
/// # Arguments
///
/// * `a` - The first number
/// * `b` - The second number
///
/// # Example
///
/// ```rust
/// use diffusionx::utils::float_eq;
///
/// let a = 1.0;
/// let b = 1.0;
/// let result = float_eq(a, b);
/// assert!(result);
/// ```
#[inline]
pub fn float_eq(a: f64, b: f64) -> bool {
    approx_eq(a, b, f64::EPSILON)
}

/// Evaluate a polynomial using the Horner method
///
/// # Arguments
///
/// * `x` - The value of the independent variable
/// * `arr` - The coefficients of the polynomial
///
/// # Example
///
/// ```
/// use diffusionx::utils::eval_poly;
///
/// let y = eval_poly(0.5, &[16., 0., 20., 0., 5., 0.]); // 6th first-kind Chebyshev polynomial
/// ```
pub fn eval_poly(x: f64, arr: &[f64]) -> f64 {
    arr.iter().fold(0.0, |acc, &a| acc * x + a)
}

/// find max value and min value in a &\[f64\]
pub fn minmax(arr: &[f64]) -> (f64, f64) {
    arr.iter()
        .copied()
        .fold((f64::MAX, f64::MIN), |(min, max), value| {
            (f64::min(min, value), f64::max(max, value))
        })
}

/// Calculate the mean and variance of an array
///
/// # Arguments
///
/// * `samples` - The array to calculate the mean and variance of
#[cfg(test)]
pub fn calculate_stats(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, variance)
}

/// Calculate the mean and variance of an integer array
///
/// # Arguments
///
/// * `samples` - The integer array to calculate the mean and variance of
#[cfg(test)]
pub fn calculate_int_stats(samples: &[u64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<u64>() as f64 / n;
    let variance = samples
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    (mean, variance)
}

/// Calculate the mean of a boolean array
///
/// # Arguments
///
/// * `samples` - The boolean array to calculate the mean of
#[cfg(test)]
pub fn calculate_bool_mean(samples: &[bool]) -> f64 {
    samples.iter().filter(|&&x| x).count() as f64 / samples.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cumsum() {
        let v = vec![1, 2, 3, 4, 5];
        let result = cumsum(0, &v);
        assert_eq!(result, vec![0, 1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_cumsum_start() {
        let v = vec![1, 2, 3, 4, 5];
        let result = cumsum(10, &v);
        assert_eq!(result, vec![10, 11, 13, 16, 20, 25]);
    }

    #[test]
    fn test_cumsum_empty() {
        let v = vec![];
        let result = cumsum(0, &v);
        assert!(result.is_empty());
    }

    #[test]
    fn test_cumsum_negative() {
        let v = vec![1, -2, 3, -4, 5];
        let result = cumsum(0, &v);
        assert_eq!(result, vec![0, 1, -1, 2, -2, 3]);
    }

    #[test]
    fn test_cumsum_float() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = cumsum(0.0, &v);
        assert_eq!(result, vec![0.0, 1.0, 3.0, 6.0, 10.0, 15.0]);
    }

    #[test]
    fn test_cumsum_negative_float() {
        let v = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let result = cumsum(0.0, &v);
        assert_eq!(result, vec![0.0, 1.0, -1.0, 2.0, -2.0, 3.0]);
    }

    #[test]
    fn test_approx_eq() {
        assert_ne!(0.1 + 0.2, 0.3);
        assert!(float_eq(0.1 + 0.2, 0.3));
    }
    #[test]
    fn test_eval_poly() {
        let arr = [
            0.3198453915289723,
            0.9076227501539942,
            0.40138509410337553,
            0.9088787482769067,
            0.7563007138750291,
        ];
        let x = 0.35625260496659283;
        let result = eval_poly(x, &arr);
        assert!(approx_eq(result, 1.1772226211231838, 1.0e-5));
        assert!(approx_eq(
            eval_poly(2.7172900350129723, &[4., 2., 9., 8.]),
            127.47717934998103,
            1.0e-5,
        ));
    }

    #[test]
    fn test_minmax() {
        let arr = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = minmax(&arr);
        assert_eq!(result, (1.0, 5.0));
    }

    #[test]
    fn test_minmax_negative() {
        let arr = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let result = minmax(&arr);
        assert_eq!(result, (-4.0, 5.0));
    }

    #[test]
    fn test_minmax_empty() {
        let arr = vec![];
        let result = minmax(&arr);
        assert_eq!(result, (f64::MAX, f64::MIN));
    }
}
