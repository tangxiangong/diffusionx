//! # Utils
//!
//! This module provides utility functions.
//!
//! ## Functions
//!
//! - `cumsum`: Calculate the cumulative sum of a vector.
//! - `gamma`: Calculate the gamma function of a number.
//! - `gammaf`: Calculate the gamma function of a number (f32 version).
//! - `approx_eq`: Check if two numbers are approximately equal.
//! - `float_eq`: Check if two numbers are equal.
//! - `eval_poly`: Evaluate a polynomial.
//! - `sinpi`: Calculate the sine of pi times a number.
//! - `cospi`: Calculate the cosine of pi times a number.
//! - `sincospi`: Calculate the sine and cosine of pi times a number.
//! - `tanpi`: Calculate the tangent of pi times a number.
//! - `minmax`: Find the minimum and maximum values in a vector.
//! - `calculate_stats`: Calculate the mean and variance of a vector, using in test.
//! - `calculate_int_stats`: Calculate the mean and variance of an integer vector, using in test.
//! - `calculate_bool_mean`: Calculate the mean of a boolean vector, using in test.

use num_traits::Num;
use std::f64::consts::PI;

/// Calculate the cumulative sum of a vector
///
/// Returns a vector of cumulative sums
///
/// # Arguments
///
/// * `start`: The initial value of the cumulative sum
/// * `v`: The vector to calculate the cumulative sum of
///
/// # Returns
///
/// Returns a vector of cumulative sums
///
/// # Example
///
/// ```rust
/// use diffusionx::utils::cumsum;
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
/// # Returns
///
/// Returns true if the two numbers are approximately equal within the tolerance, otherwise returns false.
///
/// # Example
///
/// ```rust
/// use diffusionx::utils::approx_eq;
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

/// Check if two floating numbers are equal within the f64 precision
///
/// # Arguments
///
/// * `a` - The first number
/// * `b` - The second number
///
/// # Returns
///
/// Returns true if the two numbers are equal within the f64 precision, otherwise returns false.
///
/// # Example
///
/// ```rust
/// use diffusionx::utils::float_eq;
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
/// # Returns
///
/// The value of the polynomial at `x`.
///
/// # Example
///
/// ```
/// use diffusionx::utils::eval_poly;
/// let y = eval_poly(0.5, &[16., 0., 20., 0., 5., 0.]); // 6th first-kind Chebyshev polynomial
/// ```
pub fn eval_poly(x: f64, arr: &[f64]) -> f64 {
    arr.iter().fold(0.0, |acc, &a| acc * x + a)
}

/// Calculate sin(pi * x) using the best uniform approximation polynomial, pi * x in [0, 1/4]
pub(crate) fn sinpi_kernel(x: f64) -> f64 {
    let x_square = x * x;
    let x_forth = x_square * x_square;
    let r = eval_poly(
        x,
        &[
            -2.1717412523382308e-5,
            4.662827319453555e-4,
            -7.370429884921779e-3,
            0.08214588658006512,
            -0.5992645293202981,
            2.5501640398773415,
        ],
    );
    let tmp = (-5.16771278004997f64).mul_add(x_square, x_forth.mul_add(r, 1.2245907532225998e-16));
    PI.mul_add(x, x * tmp)
}

/// Calculate cos(pi * x) using the best uniform approximation polynomial, pi * x in [0, 1/4]
pub(crate) fn cospi_kernel(x: f64) -> f64 {
    let x_square = x * x;
    let r = x_square
        * eval_poly(
            x_square,
            &[
                -1.0368935675474665e-4,
                1.9294917136379183e-3,
                -0.025806887811869204,
                0.23533063027900392,
                -1.3352627688537357,
                4.058712126416765,
            ],
        );
    let a_x_square = 4.934802200544679 * x_square;
    let a_x_square_lo = 3.109686485461973e-16f64.mul_add(
        x_square,
        4.934802200544679f64.mul_add(x_square, -a_x_square),
    );
    let w = 1.0 - a_x_square;
    w + x_square.mul_add(r, ((1.0 - w) - a_x_square) - a_x_square_lo)
}

/// Calculate sin(pi * x)
///
/// More accurate than calculating sin(pi * x), especially when x is large
///
/// If both sine and cosine values are needed, see [sincospi]
///
/// # Panic
///
/// When `x` is `f64::INFINITY` or `f64::NEG_INFINITY`, panic
///
/// # Panic
///
/// When `x` is `f64::INFINITY` or `f64::NEG_INFINITY`, panic
pub fn sinpi(_x: f64) -> f64 {
    if _x.is_nan() {
        return f64::NAN;
    }
    if _x.is_infinite() {
        panic!("function `sinpi` only accepts finite values");
    }
    let x = _x.abs();
    // 对于特别大的 x, 返回 0
    if x >= f64::MAX.floor() {
        return 0.0f64.copysign(_x);
    }

    // 根据正弦函数的周期性，将 x 转化为 [0, 1/2]
    let n = (2. * x).round();
    let rx = (-0.5f64).mul_add(n, x);
    let n = n as i64 & 3i64;
    let res = match n {
        0 => sinpi_kernel(rx),
        1 => cospi_kernel(rx),
        2 => 0.0f64 - sinpi_kernel(rx),
        _ => 0.0f64 - cospi_kernel(rx),
    };
    res.copysign(_x)
}

/// Calculate cos(pi * x)
///
/// More accurate than calculating cos(pi * x), especially when x is large
///
/// If both sine and cosine values are needed, see [sincospi]
///
/// # Panic
///
/// When `x` is `f64::INFINITY` or `f64::NEG_INFINITY`, panic
///
/// # Panic
///
/// When `x` is `f64::INFINITY` or `f64::NEG_INFINITY`, panic
pub fn cospi(_x: f64) -> f64 {
    if _x.is_nan() {
        return f64::NAN;
    }
    if _x.is_infinite() {
        panic!("function `cospi` only accepts finite values");
    }
    let x = _x.abs();
    // 对于特别大的 x, 返回 1
    if x >= f64::MAX.floor() {
        return 1.0f64.copysign(_x);
    }

    let n = (2. * x).round();
    let rx = (-0.5f64).mul_add(n, x);
    let n = n as i64 & 3i64;
    match n {
        0 => cospi_kernel(rx),
        1 => 0.0f64 - sinpi_kernel(rx),
        2 => 0.0f64 - cospi_kernel(rx),
        _ => sinpi_kernel(rx),
    }
}

/// Calculate sin(pi * x) and cos(pi * x)
///
/// Return a tuple
///
/// If only sine or cosine value is needed, see [sinpi] and [cospi]
///
/// # Panic
///
/// When `x` is `f64::INFINITY` or `f64::NEG_INFINITY`, panic
pub fn sincospi(_x: f64) -> (f64, f64) {
    if _x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if _x.is_infinite() {
        panic!("function `sincospi` only accepts finite values");
    }
    let x = _x.abs();

    if x >= f64::MAX.floor() {
        return (0.0f64.copysign(_x), 1.0f64.copysign(_x));
    }

    let n = (2. * x).round();
    let rx = (-0.5f64).mul_add(n, x);
    let n = n as i64 & 3i64;
    let si = sinpi_kernel(rx);
    let co = cospi_kernel(rx);
    match n {
        0 => (si.copysign(_x), co),
        1 => (co.copysign(_x), 0.0f64 - si),
        2 => ((0.0f64 - si).copysign(_x), 0.0f64 - co),
        _ => ((0.0f64 - co).copysign(_x), si),
    }
}

/// Calculate tan(pi * x)
///
/// More accurate than calculating tan(pi * x), especially when x is large
///
/// Similar functions: [sinpi], [cospi], [sincospi]
///
/// # Panic
///
/// When `x` is `f64::INFINITY` or `f64::NEG_INFINITY`, panic
pub fn tanpi(_x: f64) -> f64 {
    let (si, co) = sincospi(_x);
    si / co
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
///
/// # Returns
///
/// Return a tuple, containing the mean and variance
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
///
/// # Returns
///
/// Return a tuple, containing the mean and variance
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
///
/// # Returns
///
/// Return the proportion of True
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
    fn test_sinpi() {
        let tol = 1.0e-3;
        assert!(approx_eq(sinpi(1.0), 0.0, tol));
        assert!(approx_eq(sinpi(1.0 / 6.0), 0.5, tol));
    }

    #[test]
    fn test_cospi() {
        let tol = 1.0e-3;
        assert!(approx_eq(cospi(1.0), -1.0, tol));
        assert!(approx_eq(cospi(1.0 / 3.0), 0.5, tol));
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
