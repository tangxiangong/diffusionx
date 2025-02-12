//! Triangle functions
//!
//! This module provides functions for calculating the triangle function.

use super::eval_poly;
use std::f64::consts::PI;

/// 使用最佳一致逼近多项式计算 sin(pi * x), pi * x in [0, 1/4]
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

/// 使用最佳一致逼近多项式计算 cos(pi * x), pi * x in [0, 1/4]
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

/// 计算 `sin(pi x)`
///
/// 比计算 `sin(pi * x)` 更加精确, 特别是当 `x` 比较大时
///
/// /// 若同时需要正弦值和余弦值, 请见 [sincospi]
///
/// # Panic
///
/// 当 `x` 为 `f64::INFINITY` 或者 `f64::NEG_INFINITY` 时 panic
pub fn sinpi(_x: f64) -> f64 {
    if _x.is_nan() {
        return f64::NAN;
    }
    if _x.is_infinite() {
        panic!("函数 `sinpi` 只接受有限值的参数");
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

/// 计算 `cos(pi x)`
///
/// 比计算 `cos(pi * x)` 更加精确, 特别是当 `x` 比较大时
///
/// 若同时需要正弦值和余弦值, 请见 [sincospi]
///
/// # Panic
///
/// 当 `x` 为 `f64::INFINITY` 或者 `f64::NEG_INFINITY` 时 panic
pub fn cospi(_x: f64) -> f64 {
    if _x.is_nan() {
        return f64::NAN;
    }
    if _x.is_infinite() {
        panic!("函数 `cospi` 只接受有限值的参数");
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

/// 计算 `sin(pi x)` 和 `cos(pi x)`
///
/// 返回一个元组
///
/// 若只需要正弦值或者余弦值, 请见 [sinpi] 和 [cospi]
///
/// # Panic
///
/// 当 `x` 为 `f64::INFINITY` 或者 `f64::NEG_INFINITY` 时 panic
pub fn sincospi(_x: f64) -> (f64, f64) {
    if _x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if _x.is_infinite() {
        panic!("函数 `sincospi` 只接受有限值的参数");
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

/// 计算 `tan(pi x)`
///
/// 比计算 `tan(pi * x)` 更加精确, 特别是当 `x` 比较大时
///
/// 类似的函数有 [sinpi], [cospi], [sincospi]
///
/// # Panic
///
/// 当 `x` 为 `f64::INFINITY` 或者 `f64::NEG_INFINITY` 时 panic
pub fn tanpi(_x: f64) -> f64 {
    let (si, co) = sincospi(_x);
    si / co
}

#[cfg(test)]
mod tests {
    use super::super::utils::approx_eq;
    use super::*;

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
}
