//! Poisson distribution random number generation

use crate::XResult;
use rand::{prelude::*, rng};
use rand_distr::Poisson;
use rayon::prelude::*;

/// Generate a Poisson random number
///
/// This function generates a Poisson random number using the `Poisson` distribution.
///
/// # Returns
///
/// A `u64` value representing the generated random number.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::poisson::rand;
/// let random = rand(1.0);
/// ```
pub fn rand(lambda: impl Into<f64>) -> XResult<u64> {
    let lambda = lambda.into();
    let poisson = Poisson::new(lambda)?;
    Ok(rng().sample(poisson) as u64)
}

/// Generate a vector of Poisson random numbers
///
/// This function generates a vector of Poisson random numbers using the `Poisson` distribution.
///
/// # Returns
///
/// A vector of `u64` values representing the generated random numbers.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::poisson::rands;
/// let randoms = rands(1.0, 10);
/// ```
pub fn rands(lambda: impl Into<f64>, n: usize) -> XResult<Vec<u64>> {
    let lambda = lambda.into();
    let poisson = Poisson::new(lambda)?;
    Ok((0..n)
        .into_par_iter()
        .map_init(rng, |r, _| r.sample(poisson) as u64)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::calculate_int_stats;

    #[test]
    fn test_rand() {
        let _random = rand(1.0).unwrap();
    }

    #[test]
    fn test_rands() {
        let randoms = rands(1.0, 10).unwrap();
        assert_eq!(randoms.len(), 10);
    }

    #[test]
    fn test_poisson_stats() {
        let n = 1_000_000;
        let lambda = 5.0;
        let samples = rands(lambda, n).unwrap();
        let (mean, variance) = calculate_int_stats(&samples);

        assert!(
            (mean - lambda).abs() < 0.05,
            "泊松分布的均值应接近{}，实际为{}",
            lambda,
            mean
        );
        assert!(
            (variance - lambda).abs() < 0.1,
            "泊松分布的方差应接近{}，实际为{}",
            lambda,
            variance
        );
    }

    #[test]
    fn test_poisson_small_lambda_stats() {
        let n = 1_000_000;
        let lambda = 0.5;
        let samples = rands(lambda, n).unwrap();
        let (mean, variance) = calculate_int_stats(&samples);

        assert!(
            (mean - lambda).abs() < 0.02,
            "小λ值泊松分布的均值应接近{}，实际为{}",
            lambda,
            mean
        );
        assert!(
            (variance - lambda).abs() < 0.05,
            "小λ值泊松分布的方差应接近{}，实际为{}",
            lambda,
            variance
        );
    }

    #[test]
    fn test_poisson_large_lambda_stats() {
        let n = 1_000_000;
        let lambda = 50.0;
        let samples = rands(lambda, n).unwrap();
        let (mean, variance) = calculate_int_stats(&samples);

        assert!(
            (mean - lambda).abs() / lambda < 0.02,
            "大λ值泊松分布的均值应接近{}，实际为{}",
            lambda,
            mean
        );
        assert!(
            (variance - lambda).abs() / lambda < 0.05,
            "大λ值泊松分布的方差应接近{}，实际为{}",
            lambda,
            variance
        );
    }
}
