//! Exponential distribution random number generation

use crate::{XError, XResult};
use num_traits::float::Float;
use rand::{prelude::*, rng};
use rand_distr::{Exp, Exp1};
use rayon::prelude::*;

/// Exponential distribution
#[derive(Debug, Clone)]
pub struct Exponential<T: Float + Send + Sync = f64>
where
    Exp1: Distribution<T>,
{
    /// rate parameter
    lambda: T,
}

/// Default value for the exponential distribution
impl Default for Exponential {
    fn default() -> Self {
        Self { lambda: 1.0 }
    }
}

impl<T: Float + Send + Sync> Exponential<T>
where
    Exp1: Distribution<T>,
{
    /// Create a new exponential distribution with a given rate parameter
    ///
    /// # Arguments
    ///
    /// * `lambda` - The rate parameter of the exponential distribution, must be greater than 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::random::exponential::Exponential;
    ///
    /// let exp = Exponential::new(2.0).unwrap();
    /// ```
    pub fn new(lambda: T) -> XResult<Self>
    where
        T: std::fmt::Display,
    {
        if lambda <= T::zero() {
            return Err(XError::InvalidParameters(format!(
                "The rate parameter `lambda` must be greater than 0, got {lambda}"
            )));
        }
        Ok(Self { lambda })
    }

    /// Get the rate parameter
    pub fn get_lambda(&self) -> T {
        self.lambda
    }

    /// Generate a vector of exponential random numbers
    ///
    /// # Arguments
    ///
    /// * `n` - The number of random numbers to generate, must be greater than 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::random::exponential::Exponential;
    ///
    /// let exp = Exponential::new(2.0).unwrap();
    /// let randoms = exp.samples(10).unwrap();
    /// ```
    pub fn samples(&self, n: usize) -> XResult<Vec<T>> {
        if self.lambda == T::one() {
            Ok(standard_rands(n))
        } else {
            rands(self.lambda, n)
        }
    }
}

/// Generate a standard exponential random number
///
/// # Example
///
/// ```rust
/// use diffusionx::random::exponential::standard_rand;
///
/// let random = standard_rand::<f64>();
/// ```
pub fn standard_rand<T: Float + Send + Sync>() -> T
where
    Exp1: Distribution<T>,
{
    rng().sample(Exp1)
}

/// Generate a vector of standard exponential random numbers
///
/// # Arguments
///
/// * `n` - The number of random numbers to generate, must be greater than 0.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::exponential::standard_rands;
///
/// let randoms = standard_rands::<f64>(10);
/// ```
pub fn standard_rands<T: Float + Send + Sync>(n: usize) -> Vec<T>
where
    Exp1: Distribution<T>,
{
    let dist = Exp1;
    (0..n)
        .into_par_iter()
        .map_init(rng, |r, _| r.sample(dist))
        .collect()
}

/// Generate an exponential random number
///
/// # Arguments
///
/// * `lambda` - The rate parameter of the exponential distribution, must be greater than 0.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::exponential::rand;
///
/// let random = rand(1.0).unwrap();
/// ```
pub fn rand<T: Float + Send + Sync>(lambda: T) -> XResult<T>
where
    Exp1: Distribution<T>,
{
    let exp = Exp::new(lambda)?;
    Ok(rng().sample(exp))
}

/// Generate a vector of exponential random numbers
///
/// # Arguments
///
/// * `lambda` - The rate parameter of the exponential distribution, must be greater than 0.
/// * `n` - The number of random numbers to generate, must be greater than 0.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::exponential::rands;
///
/// let randoms = rands(1.0, 10).unwrap();
/// ```
pub fn rands<T: Float + Send + Sync>(lambda: T, n: usize) -> XResult<Vec<T>>
where
    Exp1: Distribution<T>,
{
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
        let random = standard_rand::<f64>();
        assert!(random.is_finite());
    }

    #[test]
    fn test_standard_rands() {
        let randoms = standard_rands::<f64>(10);
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
        let samples = standard_rands::<f64>(n);
        let (mean, variance) = calculate_stats(&samples);

        assert!(
            (mean - 1.0).abs() < 0.01,
            "The mean of the standard exponential distribution should be close to 1, but got {mean}"
        );
        assert!(
            (variance - 1.0).abs() < 0.05,
            "The variance of the standard exponential distribution should be close to 1, but got {variance}"
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
            "The mean of the exponential distribution should be close to {expected_mean}, but got {mean}"
        );
        assert!(
            (variance - expected_variance).abs() < 0.05,
            "The variance of the exponential distribution should be close to {expected_variance}, but got {variance}"
        );
    }
}
