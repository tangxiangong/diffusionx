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

use crate::{FloatExt, XError, XResult};
use num_traits::Num;
#[cfg(feature = "visualize")]
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
pub fn approx_eq<T: FloatExt>(a: T, b: T, tol: T) -> bool {
    if a.is_infinite() || b.is_infinite() {
        false
    } else {
        (a - b).abs() <= tol
    }
}

#[cfg(feature = "visualize")]
/// Ensure the output directory exists, or create it if it doesn't exist.
pub(crate) fn ensure_output_dir(path: &Path) -> XResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| XError::Other(e.to_string()))?;
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
pub fn float_eq<T: FloatExt>(a: T, b: T) -> bool {
    approx_eq(a, b, T::epsilon())
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
pub fn minmax<T: FloatExt>(arr: &[T]) -> (T, T) {
    arr.iter()
        .copied()
        .fold((T::max_value(), T::min_value()), |(min, max), value| {
            (T::min(min, value), T::max(max, value))
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
pub fn calculate_int_stats(samples: &[usize]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<usize>() as f64 / n;
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

/// Generate a vector of evenly spaced numbers over a specified range
///
/// This function generates a sequence from `start` to `end` (inclusive) with the given `step` size.
/// The endpoint is always included if it can be reached exactly or if the last step would overshoot it.
///
/// # Arguments
///
/// * `start` - The starting value of the range
/// * `end` - The ending value of the range (inclusive)
/// * `step` - The step size between numbers (must be positive)
///
/// # Returns
///
/// A vector containing the evenly spaced values
///
/// # Panics
///
/// Panics if `step` is not positive or if `start > end`
///
/// # Example
///
/// ```rust
/// use diffusionx::utils::linspace;
///
/// let result = linspace(0.0, 1.0, 0.25);
/// assert_eq!(result, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
/// ```
pub fn linspace<T: FloatExt>(start: T, end: T, step: T) -> Vec<T> {
    if step <= T::zero() {
        panic!("step must be positive, got {step:?}");
    }
    if start > end {
        panic!("start must be <= end, got start={start:?}, end={end:?}");
    }

    let len = ((end - start) / step).ceil().to_usize().unwrap() + 1;
    let mut result = (0..len)
        .map(|i| start + T::from(i).unwrap() * step)
        .collect::<Vec<_>>();

    let last = match result.last_mut() {
        Some(last) => last,
        None => panic!("The length of the result is 0"),
    };
    *last = end;
    result
}

/// Calculate the difference between adjacent elements in an array
///
/// # Arguments
///
/// * `arr` - The input array
pub fn diff<T>(arr: &[T]) -> Vec<T>
where
    T: Num + Copy,
{
    if arr.len() < 2 {
        return arr.to_vec();
    }
    arr.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Check if an array is non-decreasing
///
/// # Arguments
///
/// * `arr` - The input array
///
/// # Returns
///
/// `true` if the array is non-decreasing, `false` otherwise
pub fn is_increasing<T: FloatExt>(arr: &[T]) -> bool {
    arr.windows(2).all(|w| w[0] < w[1])
}

/// Linear interpolation
///
/// # Arguments
///
/// * `t` - The time points (must be strictly monotonically increasing)
/// * `x` - The corresponding values
/// * `step` - The step size for the output time sequence (must be positive)
pub fn linear_interpolate<T: FloatExt>(t: &[T], x: &[T], step: T) -> XResult<(Vec<T>, Vec<T>)> {
    if t.len() != x.len() {
        return Err(XError::Other(
            "t and x must have the same length".to_string(),
        ));
    }

    if t.len() < 2 {
        return Err(XError::Other(
            "t and x must have at least 2 elements".to_string(),
        ));
    }

    if step <= T::zero() {
        return Err(XError::Other("step must be positive".to_string()));
    }

    if !is_increasing(t) {
        return Err(XError::InvalidParameters(
            "t must be strictly monotonically increasing".to_string(),
        ));
    }

    let t_new = linspace(t[0], t[t.len() - 1], step);
    let mut x_new = Vec::with_capacity(t_new.len());

    for &t_val in &t_new {
        // 使用二分搜索找到 t_val 所在的区间
        let j = match t.binary_search_by(|&probe| probe.partial_cmp(&t_val).unwrap()) {
            Ok(exact_idx) => {
                // t_val 正好等于某个时间点
                x_new.push(x[exact_idx]);
                continue;
            }
            Err(insert_idx) => {
                if insert_idx == 0 {
                    // t_val 小于所有时间点，使用第一个值
                    x_new.push(x[0]);
                    continue;
                } else if insert_idx >= t.len() {
                    // t_val 大于所有时间点，使用最后一个值
                    x_new.push(x[t.len() - 1]);
                    continue;
                } else {
                    insert_idx - 1
                }
            }
        };

        // 线性插值: x = x[j] + (x[j+1] - x[j]) * (t_val - t[j]) / (t[j+1] - t[j])
        let ratio = (t_val - t[j]) / (t[j + 1] - t[j]);
        let x_interpolated = x[j] + (x[j + 1] - x[j]) * ratio;
        x_new.push(x_interpolated);
    }

    Ok((t_new, x_new))
}

/// Generate a flattened (step function) interpolation over a specified range
///
/// This function generates the same time sequence as `linear_interpolate`, but instead of
/// linear interpolation, it creates a left-continuous step function.
///
/// # Arguments
///
/// * `t` - The time points (must be strictly monotonically increasing)
/// * `x` - The corresponding values
/// * `step` - The step size for the output time sequence (must be positive)
pub fn flatten_interpolate(t: &[f64], x: &[f64], step: f64) -> XResult<(Vec<f64>, Vec<f64>)> {
    if t.len() != x.len() {
        return Err(XError::Other(
            "t and x must have the same length".to_string(),
        ));
    }

    if t.len() < 2 {
        return Err(XError::Other(
            "t and x must have at least 2 elements".to_string(),
        ));
    }

    if step <= 0.0 {
        return Err(XError::Other("step must be positive".to_string()));
    }

    if !is_increasing(t) {
        return Err(XError::InvalidParameters(
            "t must be strictly monotonically increasing".to_string(),
        ));
    }

    let t_new = linspace(t[0], t[t.len() - 1], step);
    let mut x_new = Vec::with_capacity(t_new.len());

    for &t_val in &t_new {
        // 使用二分搜索找到 t_val 所在的区间
        let j = match t.binary_search_by(|&probe| probe.partial_cmp(&t_val).unwrap()) {
            Ok(exact_idx) => {
                // t_val 正好等于某个时间点，使用该点的值
                x_new.push(x[exact_idx]);
                continue;
            }
            Err(insert_idx) => {
                if insert_idx == 0 {
                    // t_val 小于所有时间点，使用第一个值
                    x_new.push(x[0]);
                    continue;
                } else if insert_idx >= t.len() {
                    // t_val 大于所有时间点，使用最后一个值
                    x_new.push(x[t.len() - 1]);
                    continue;
                } else {
                    insert_idx - 1
                }
            }
        };

        // 左连续阶梯函数：使用 x[j] (区间 [t[j], t[j+1]) 的左端点值)
        x_new.push(x[j]);
    }

    Ok((t_new, x_new))
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

    #[test]
    fn test_linspace() {
        // 基本测试
        let result = linspace(0.0, 1.0, 0.25);
        let expected = [0.0, 0.25, 0.5, 0.75, 1.0];
        assert_eq!(result.len(), expected.len());
        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert!(approx_eq(*actual, *expected, 1e-10));
        }

        // 测试不能整除的情况
        let result = linspace(0.0, 1.0, 0.3);
        assert!(result.contains(&0.0));
        assert!(result.contains(&1.0)); // 应该包含终点
        assert!(result.iter().any(|&x| approx_eq(x, 0.3, 1e-10)));
        assert!(result.iter().any(|&x| approx_eq(x, 0.6, 1e-10)));
        assert!(result.iter().any(|&x| approx_eq(x, 0.9, 1e-10)));

        // 测试单点情况
        let result = linspace(5.0, 5.0, 0.1);
        assert_eq!(result, vec![5.0]);

        // 测试小范围
        let result = linspace(0.0, 0.1, 0.05);
        let expected = [0.0, 0.05, 0.1];
        assert_eq!(result.len(), expected.len());
        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert!(approx_eq(*actual, *expected, 1e-10));
        }
    }

    #[test]
    #[should_panic(expected = "step must be positive")]
    fn test_linspace_negative_step() {
        linspace(0.0, 1.0, -0.1);
    }

    #[test]
    #[should_panic(expected = "step must be positive")]
    fn test_linspace_zero_step() {
        linspace(0.0, 1.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "start must be <= end")]
    fn test_linspace_invalid_range() {
        linspace(1.0, 0.0, 0.1);
    }

    use crate::simulation::continuous::levy_walk::simulate_levy_walk_with_duration;
    #[test]
    fn test_interpolate() {
        let (t, x) = simulate_levy_walk_with_duration(1.0, 1.0, 10.0, 0.0).unwrap();
        println!("t: {t:?}, x: {x:?}");
        let result = linear_interpolate(&t, &x, 0.1).unwrap();
        println!("result: {result:?}");
    }

    #[test]
    fn test_linear_interpolate_simple() {
        // 简单的测试数据：(0,0), (1,1), (2,4)
        let t = vec![0.0, 1.0, 2.0];
        let x = vec![0.0, 1.0, 4.0];

        let (t_new, x_new) = linear_interpolate(&t, &x, 0.5).unwrap();

        // 期望的结果：t_new = [0.0, 0.5, 1.0, 1.5, 2.0]
        // x_new = [0.0, 0.5, 1.0, 2.5, 4.0]
        let expected_t = [0.0, 0.5, 1.0, 1.5, 2.0];
        let expected_x = [0.0, 0.5, 1.0, 2.5, 4.0];

        assert_eq!(t_new.len(), expected_t.len());
        assert_eq!(x_new.len(), expected_x.len());

        for (i, (&actual_t, &expected_t)) in t_new.iter().zip(expected_t.iter()).enumerate() {
            assert!(
                approx_eq(actual_t, expected_t, 1e-10),
                "t_new[{i}]: expected {expected_t}, got {actual_t}"
            );
        }

        for (i, (&actual_x, &expected_x)) in x_new.iter().zip(expected_x.iter()).enumerate() {
            assert!(
                approx_eq(actual_x, expected_x, 1e-10),
                "x_new[{i}]: expected {expected_x}, got {actual_x}"
            );
        }
    }

    #[test]
    fn test_linear_interpolate_edge_cases() {
        // 测试步长为负数的情况
        let t = vec![0.0, 1.0, 2.0];
        let x = vec![0.0, 1.0, 4.0];
        assert!(linear_interpolate(&t, &x, -0.1).is_err());

        // 测试步长为零的情况
        assert!(linear_interpolate(&t, &x, 0.0).is_err());

        // 测试非单调递增的时间序列
        let t_bad = vec![0.0, 2.0, 1.0];
        let x_bad = vec![0.0, 1.0, 4.0];
        assert!(linear_interpolate(&t_bad, &x_bad, 0.1).is_err());

        // 测试长度不匹配
        let t_short = vec![0.0, 1.0];
        let x_long = vec![0.0, 1.0, 4.0];
        assert!(linear_interpolate(&t_short, &x_long, 0.1).is_err());

        // 测试数据点太少
        let t_single = vec![0.0];
        let x_single = vec![0.0];
        assert!(linear_interpolate(&t_single, &x_single, 0.1).is_err());
    }

    #[test]
    fn test_linear_interpolate_boundary() {
        // 测试边界情况：插值点超出原始数据范围
        let t = vec![1.0, 2.0, 3.0];
        let x = vec![10.0, 20.0, 30.0];

        // 使用更大的步长，使得插值点包含边界
        let (t_new, x_new) = linear_interpolate(&t, &x, 0.5).unwrap();

        // 第一个点应该是 (1.0, 10.0)
        assert!(approx_eq(t_new[0], 1.0, 1e-10));
        assert!(approx_eq(x_new[0], 10.0, 1e-10));

        // 最后一个点应该是 (3.0, 30.0)
        let last_idx = t_new.len() - 1;
        assert!(approx_eq(t_new[last_idx], 3.0, 1e-10));
        assert!(approx_eq(x_new[last_idx], 30.0, 1e-10));

        // 中间点 (1.5, 15.0) 和 (2.5, 25.0) 应该通过线性插值得到
        let mid1_idx = t_new
            .iter()
            .position(|&t| approx_eq(t, 1.5, 1e-10))
            .unwrap();
        assert!(approx_eq(x_new[mid1_idx], 15.0, 1e-10));

        let mid2_idx = t_new
            .iter()
            .position(|&t| approx_eq(t, 2.5, 1e-10))
            .unwrap();
        assert!(approx_eq(x_new[mid2_idx], 25.0, 1e-10));
    }

    #[test]
    fn test_flatten_interpolate() {
        // 简单的测试数据：(0,10), (1,20), (2,30)
        let t = vec![0.0, 1.0, 2.0];
        let x = vec![10.0, 20.0, 30.0];

        let (t_new, x_new) = flatten_interpolate(&t, &x, 0.5).unwrap();

        // 期望的结果：
        // t_new = [0.0, 0.5, 1.0, 1.5, 2.0]
        // x_new = [10, 10, 20, 20, 30]  (左连续阶梯)
        let expected_t = [0.0, 0.5, 1.0, 1.5, 2.0];
        let expected_x = [10.0, 10.0, 20.0, 20.0, 30.0];

        assert_eq!(t_new.len(), expected_t.len());
        assert_eq!(x_new.len(), expected_x.len());

        for (i, (&actual_t, &expected_t)) in t_new.iter().zip(expected_t.iter()).enumerate() {
            assert!(
                approx_eq(actual_t, expected_t, 1e-10),
                "t_new[{i}]: expected {expected_t}, got {actual_t}"
            );
        }

        for (i, (&actual_x, &expected_x)) in x_new.iter().zip(expected_x.iter()).enumerate() {
            assert!(
                approx_eq(actual_x, expected_x, 1e-10),
                "x_new[{i}]: expected {expected_x}, got {actual_x}"
            );
        }
    }

    #[test]
    fn test_flatten_vs_linear_interpolate_time_consistency() {
        // 验证 flatten_interpolate 和 linear_interpolate 的时间序列一致
        let t = vec![0.0, 1.5, 3.2, 5.0];
        let x = vec![100.0, 200.0, 150.0, 300.0];
        let step = 0.3;

        let (t_linear, _) = linear_interpolate(&t, &x, step).unwrap();
        let (t_flatten, _) = flatten_interpolate(&t, &x, step).unwrap();

        assert_eq!(t_linear.len(), t_flatten.len());
        for (i, (&t_lin, &t_flat)) in t_linear.iter().zip(t_flatten.iter()).enumerate() {
            assert!(
                approx_eq(t_lin, t_flat, 1e-10),
                "Time sequences differ at index {i}: linear={t_lin}, flatten={t_flat}"
            );
        }
    }

    #[test]
    fn test_linear_vs_flatten_comparison() {
        // 详细对比 linear_interpolate 和 flatten_interpolate 的输出
        let t = vec![0.0, 1.0, 2.0];
        let x = vec![10.0, 20.0, 30.0];
        let step = 0.5;

        let (t_linear, x_linear) = linear_interpolate(&t, &x, step).unwrap();
        let (t_flatten, x_flatten) = flatten_interpolate(&t, &x, step).unwrap();

        println!("原始数据: t={t:?}, x={x:?}");
        println!("线性插值: t_new={t_linear:?}, x_new={x_linear:?}");
        println!("阶梯插值: t_new={t_flatten:?}, x_new={x_flatten:?}");

        // 验证时间序列一致
        assert_eq!(t_linear, t_flatten);

        // 验证值序列的差异
        // 在 t=0.0: 两者都应该是 10.0
        assert!(approx_eq(x_linear[0], 10.0, 1e-10));
        assert!(approx_eq(x_flatten[0], 10.0, 1e-10));

        // 在 t=0.5: linear 应该是 15.0 (插值), flatten 应该是 10.0 (左连续)
        assert!(approx_eq(x_linear[1], 15.0, 1e-10));
        assert!(approx_eq(x_flatten[1], 10.0, 1e-10));

        // 在 t=1.0: 两者都应该是 20.0
        assert!(approx_eq(x_linear[2], 20.0, 1e-10));
        assert!(approx_eq(x_flatten[2], 20.0, 1e-10));

        // 在 t=1.5: linear 应该是 25.0 (插值), flatten 应该是 20.0 (左连续)
        assert!(approx_eq(x_linear[3], 25.0, 1e-10));
        assert!(approx_eq(x_flatten[3], 20.0, 1e-10));

        // 在 t=2.0: 两者都应该是 30.0
        assert!(approx_eq(x_linear[4], 30.0, 1e-10));
        assert!(approx_eq(x_flatten[4], 30.0, 1e-10));
    }
}
