//! Poisson distribution random number generation

use crate::{FloatExt, IntExt, XError, XResult};
use rand::prelude::*;
use rand_distr::{Exp1, StandardNormal, StandardUniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

#[derive(Debug, Clone)]
/// Poisson distribution
pub struct Poisson<T: FloatExt = f64> {
    /// rate parameter, must be greater than 0
    lambda: T,
}

impl<T: FloatExt> Default for Poisson<T> {
    fn default() -> Self {
        Self { lambda: T::one() }
    }
}

impl<T: FloatExt> Poisson<T> {
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
    pub fn new(lambda: T) -> XResult<Self> {
        if lambda <= T::zero() {
            return Err(XError::InvalidParameters(format!(
                "The rate parameter `lambda` must be greater than 0, got {lambda:?}"
            )));
        }
        Ok(Self { lambda })
    }

    /// Get the rate parameter
    pub fn get_lambda(&self) -> T {
        self.lambda
    }

    /// Generate a vector of Poisson random numbers
    pub fn samples<U: IntExt>(&self, n: usize) -> XResult<Vec<U>>
    where
        StandardUniform: Distribution<T>,
        Exp1: Distribution<T>,
        StandardNormal: Distribution<T>,
    {
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
pub fn rand<T: FloatExt, U: IntExt>(lambda: T) -> XResult<U>
where
    StandardUniform: Distribution<T>,
    Exp1: Distribution<T>,
    StandardNormal: Distribution<T>,
{
    let poisson = rand_distr::Poisson::new(lambda)?;
    let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
    Ok(U::from(rng.sample(poisson)).unwrap())
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
pub fn rands<T: FloatExt, U: IntExt>(lambda: T, n: usize) -> XResult<Vec<U>>
where
    StandardUniform: Distribution<T>,
    Exp1: Distribution<T>,
    StandardNormal: Distribution<T>,
{
    let poisson = rand_distr::Poisson::new(lambda)?;
    Ok((0..n)
        .into_par_iter()
        .map_init(
            || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
            |r, _| U::from(r.sample(poisson)).unwrap(),
        )
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::calculate_int_stats;

    #[test]
    fn test_rand() {
        let _random = rand::<_, usize>(1.0).unwrap();
    }

    #[test]
    fn test_rands() {
        let randoms = rands::<_, usize>(1.0, 10).unwrap();
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
            "The mean of the Poisson distribution should be close to {lambda}, got {mean}"
        );
        assert!(
            (variance - lambda).abs() < 0.1,
            "The variance of the Poisson distribution should be close to {lambda}, got {variance}"
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
            "The mean of the Poisson distribution should be close to {lambda}, got {mean}"
        );
        assert!(
            (variance - lambda).abs() < 0.05,
            "The variance of the Poisson distribution should be close to {lambda}, got {variance}"
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
            "The mean of the Poisson distribution should be close to {lambda}, got {mean}"
        );
        assert!(
            (variance - lambda).abs() / lambda < 0.05,
            "The variance of the Poisson distribution should be close to {lambda}, got {variance}"
        );
    }
}
