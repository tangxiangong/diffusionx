//! Poisson distribution random number generation
//!

use crate::{XError, XResult};
use rand::{prelude::*, rng};
use rayon::prelude::*;

/// Poisson distribution
pub struct Poisson {
    /// rate parameter, must be greater than 0
    lambda: f64,
}

impl Default for Poisson {
    fn default() -> Self {
        Self { lambda: 1.0 }
    }
}

impl Poisson {
    /// Create a new Poisson distribution with a given rate parameter
    ///
    /// # Arguments
    ///
    /// * `lambda` - The rate parameter of the Poisson distribution, must be greater than 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::random::poisson::Poisson;
    ///
    /// let lambda = 1.0;
    /// let poisson = Poisson::new(lambda).unwrap();
    /// ```
    pub fn new(lambda: impl Into<f64>) -> XResult<Self> {
        let lambda = lambda.into();
        if lambda <= 0.0 {
            return Err(XError::InvalidParameters(format!(
                "The rate parameter `lambda` must be greater than 0, got {}",
                lambda
            )));
        }
        Ok(Self { lambda })
    }

    /// Get the rate parameter
    pub fn get_lambda(&self) -> f64 {
        self.lambda
    }

    /// Generate a vector of Poisson random numbers
    pub fn samples(&self, n: usize) -> XResult<Vec<usize>> {
        rands(self.lambda, n)
    }
}

/// Generate a Poisson random number
///
/// # Arguments
///
/// * `lambda` - The rate parameter of the Poisson distribution, must be greater than 0.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::poisson::rand;
///
/// let random = rand(1.0).unwrap();
/// ```
pub fn rand(lambda: impl Into<f64>) -> XResult<usize> {
    let lambda: f64 = lambda.into();
    let poisson = rand_distr::Poisson::new(lambda)?;
    Ok(rng().sample(poisson) as usize)
}

/// Generate a vector of Poisson random numbers
///
/// # Arguments
///
/// * `lambda` - The rate parameter of the Poisson distribution, must be greater than 0.
/// * `n` - The number of random numbers to generate, must be greater than 0.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::poisson::rands;
///
/// let randoms = rands(1.0, 10).unwrap();
/// ```
pub fn rands(lambda: impl Into<f64>, n: usize) -> XResult<Vec<usize>> {
    let lambda: f64 = lambda.into();
    let poisson = rand_distr::Poisson::new(lambda)?;
    Ok((0..n)
        .into_par_iter()
        .map_init(rng, |r, _| r.sample(poisson) as usize)
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
            "The mean of the Poisson distribution should be close to {}, got {}",
            lambda,
            mean
        );
        assert!(
            (variance - lambda).abs() < 0.1,
            "The variance of the Poisson distribution should be close to {}, got {}",
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
            "The mean of the Poisson distribution should be close to {}, got {}",
            lambda,
            mean
        );
        assert!(
            (variance - lambda).abs() < 0.05,
            "The variance of the Poisson distribution should be close to {}, got {}",
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
            "The mean of the Poisson distribution should be close to {}, got {}",
            lambda,
            mean
        );
        assert!(
            (variance - lambda).abs() / lambda < 0.05,
            "The variance of the Poisson distribution should be close to {}, got {}",
            lambda,
            variance
        );
    }
}
