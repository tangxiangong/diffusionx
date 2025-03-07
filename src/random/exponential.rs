//! Exponential distribution random number generation

use crate::XResult;
use rand::{prelude::*, rng};
use rand_distr::{Exp, Exp1};
use rayon::prelude::*;

/// Generate a standard exponential random number
///
/// This function generates a standard exponential random number using the `Exp1` distribution.
///
/// # Returns
///
/// A `f64` value representing the generated random number.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::exponential::standard_rand;
/// let random = standard_rand();
/// ```
pub fn standard_rand() -> f64 {
    rng().sample(Exp1)
}

/// Generate a vector of standard exponential random numbers
///
/// This function generates a vector of standard exponential random numbers using the `Exp1` distribution.
///
/// # Returns
///
/// A vector of `f64` values representing the generated random numbers.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::exponential::standard_rands;
/// let randoms = standard_rands(10);
/// ```
pub fn standard_rands(n: usize) -> Vec<f64> {
    let dist = Exp1;
    (0..n)
        .into_par_iter()
        .map_init(rng, |r, _| r.sample(dist))
        .collect()
}

/// Generate an exponential random number
///
/// This function generates an exponential random number using the `Exp` distribution.
///
/// # Returns
///
/// A `f64` value representing the generated random number.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::exponential::rand;
/// let random = rand(1.0);
/// ```
pub fn rand(lambda: impl Into<f64>) -> XResult<f64> {
    let lambda = lambda.into();
    let exp = Exp::new(lambda)?;
    Ok(rng().sample(exp))
}

/// Generate a vector of exponential random numbers
///
/// This function generates a vector of exponential random numbers using the `Exp` distribution.
///
/// # Returns
///
/// A vector of `f64` values representing the generated random numbers.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::exponential::rands;
/// let randoms = rands(1.0, 10);
/// ```
pub fn rands(lambda: impl Into<f64>, n: usize) -> XResult<Vec<f64>> {
    let lambda = lambda.into();
    let exp = Exp::new(lambda)?;
    Ok((0..n)
        .into_par_iter()
        .map_init(rng, |r, _| r.sample(exp))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::calculate_stats;

    #[test]
    fn test_standard_rand() {
        let random = standard_rand();
        assert!(random.is_finite());
    }

    #[test]
    fn test_standard_rands() {
        let randoms = standard_rands(10);
        assert_eq!(randoms.len(), 10);
        assert!(randoms.iter().all(|r| r.is_finite()));
    }

    #[test]
    fn test_rand() {
        let random = rand(1.0).unwrap();
        assert!(random.is_finite());
    }

    #[test]
    fn test_rands() {
        let randoms = rands(1.0, 10).unwrap();
        assert_eq!(randoms.len(), 10);
        assert!(randoms.iter().all(|r| r.is_finite()));
    }

    #[test]
    fn test_standard_exponential_stats() {
        let n = 1_000_000;
        let samples = standard_rands(n);
        let (mean, variance) = calculate_stats(&samples);

        assert!(
            (mean - 1.0).abs() < 0.01,
            "标准指数分布的均值应接近1，实际为{}",
            mean
        );
        assert!(
            (variance - 1.0).abs() < 0.05,
            "标准指数分布的方差应接近1，实际为{}",
            variance
        );
    }

    #[test]
    fn test_exponential_stats() {
        let n = 1_000_000;
        let lambda = 2.0;
        let samples = rands(lambda, n).unwrap();
        let (mean, variance) = calculate_stats(&samples);

        let expected_mean = 1.0 / lambda;
        let expected_variance = 1.0 / (lambda * lambda);

        assert!(
            (mean - expected_mean).abs() < 0.01,
            "指数分布的均值应接近{}，实际为{}",
            expected_mean,
            mean
        );
        assert!(
            (variance - expected_variance).abs() < 0.05,
            "指数分布的方差应接近{}，实际为{}",
            expected_variance,
            variance
        );
    }
}
