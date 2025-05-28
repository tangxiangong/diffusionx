//! Normal random number generation
//! For other stable distributions, see [crate::random::stable].

use crate::{XError, XResult};
use num_traits::float::Float;
use rand::{prelude::*, rng};
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::ops::{Add, Mul, Neg, Sub};

/// Normal distribution
#[derive(Debug, Clone)]
pub struct Normal<T: Float + Send + Sync = f64> {
    /// mean
    mu: T,
    /// standard deviation
    sigma: T,
}

impl Default for Normal {
    fn default() -> Self {
        Self {
            mu: 0.0,
            sigma: 1.0,
        }
    }
}

impl<T: Float + Send + Sync> Normal<T> {
    /// Create a new normal distribution with a given mean and standard deviation
    ///
    /// # Arguments
    ///
    /// * `mu` - The mean of the normal distribution.
    /// * `sigma` - The standard deviation of the normal distribution, must be greater than 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::random::normal::Normal;
    ///
    /// let mu = 1.0;
    /// let sigma = 2.0;
    /// let normal = Normal::new(mu, sigma).unwrap();
    /// ```
    pub fn new(mu: T, sigma: T) -> XResult<Self>
    where
        T: std::fmt::Display,
    {
        if sigma <= T::zero() {
            return Err(XError::InvalidParameters(format!(
                "The standard deviation `sigma` must be greater than 0, got {}",
                sigma
            )));
        }
        Ok(Self { mu, sigma })
    }

    /// Get the mean
    pub fn get_mu(&self) -> T {
        self.mu
    }

    /// Get the standard deviation
    pub fn get_sigma(&self) -> T {
        self.sigma
    }

    /// Generate a vector of normal random numbers
    ///
    /// # Arguments
    ///
    /// * `n` - The number of random numbers to generate, must be greater than 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::random::normal::Normal;
    ///
    /// let normal = Normal::default();
    /// let randoms = normal.samples(10).unwrap();
    /// ```
    pub fn samples(&self, n: usize) -> XResult<Vec<T>>
    where
        StandardNormal: Distribution<T>,
    {
        if self.sigma == T::one() && self.mu == T::zero() {
            Ok(standard_rands(n))
        } else {
            rands(self.mu, self.sigma, n)
        }
    }
}

impl<T: Float + Send + Sync> Neg for Normal<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            mu: -self.mu,
            sigma: self.sigma,
        }
    }
}

impl<T: Float + Send + Sync> Add for Normal<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let sigma = (self.sigma * self.sigma + rhs.sigma * rhs.sigma).sqrt();
        Self {
            mu: self.mu + rhs.mu,
            sigma,
        }
    }
}

impl<T: Float + Send + Sync> Sub for Normal<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<T: Float + Send + Sync> Mul<T> for Normal<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            mu: self.mu * rhs,
            sigma: self.sigma * rhs.abs(),
        }
    }
}

impl Mul<Normal> for f64 {
    type Output = Normal;

    fn mul(self, rhs: Normal) -> Self::Output {
        Self::Output {
            mu: self * rhs.mu,
            sigma: self.abs() * rhs.sigma,
        }
    }
}

impl<T: Send + Sync + Float> Add<T> for Normal<T> {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Self {
            mu: self.mu + rhs,
            sigma: self.sigma,
        }
    }
}

impl Add<Normal> for f64 {
    type Output = Normal;

    fn add(self, rhs: Normal) -> Self::Output {
        Self::Output {
            mu: self + rhs.mu,
            sigma: rhs.sigma,
        }
    }
}

impl<T: Send + Sync + Float> Sub<T> for Normal<T> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        self + (-rhs)
    }
}

impl Sub<Normal> for f64 {
    type Output = Normal;

    fn sub(self, rhs: Normal) -> Self::Output {
        self + (-rhs)
    }
}

/// Generate a standard normal random number
///
/// # Example
///
/// ```rust
/// use diffusionx::random::normal::standard_rand;
///
/// let random = standard_rand::<f64>();
/// ```
pub fn standard_rand<T: Float + Send + Sync>() -> T
where
    StandardNormal: Distribution<T>,
{
    rng().sample(rand_distr::StandardNormal)
}

/// Generate a vector of standard normal random numbers
///
/// # Example
///
/// ```rust
/// use diffusionx::random::normal::standard_rands;
///
/// let randoms = standard_rands::<f64>(10);
/// ```
pub fn standard_rands<T: Float + Send + Sync>(n: usize) -> Vec<T>
where
    StandardNormal: Distribution<T>,
{
    let dist = rand_distr::StandardNormal;
    (0..n)
        .into_par_iter()
        .map_init(rng, |r, _| r.sample(dist))
        .collect()
}

/// Generate a normal random number
///
/// # Arguments
///
/// * `mean` - The mean of the normal distribution.
/// * `std_dev` - The standard deviation of the normal distribution, must be greater than 0.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::normal::rand;
///
/// let random = rand(0.0, 1.0).unwrap();
/// ```
pub fn rand<T: Float + Send + Sync>(mean: T, std_dev: T) -> XResult<T>
where
    StandardNormal: Distribution<T>,
{
    let normal = rand_distr::Normal::new(mean, std_dev)?;
    Ok(rng().sample(normal))
}

/// Generate a vector of normal random numbers
///
/// # Arguments
///
/// * `mean` - The mean of the normal distribution.
/// * `std_dev` - The standard deviation of the normal distribution, must be greater than 0.
/// * `n` - The number of random numbers to generate, must be greater than 0.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::normal::rands;
///
/// let randoms = rands(0.0, 1.0, 10).unwrap();
/// ```
pub fn rands<T: Float + Send + Sync>(mean: T, std_dev: T, n: usize) -> XResult<Vec<T>>
where
    StandardNormal: Distribution<T>,
{
    let normal = rand_distr::Normal::new(mean, std_dev)?;
    Ok((0..n)
        .into_par_iter()
        .map_init(rng, |r, _| r.sample(normal))
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
        let random = rand(0.0, 1.0).unwrap();
        assert!(random.is_finite());
    }

    #[test]
    fn test_rands() {
        let randoms = rands(0.0, 1.0, 10).unwrap();
        assert_eq!(randoms.len(), 10);
        assert!(randoms.iter().all(|r| r.is_finite()));
    }

    #[test]
    fn test_standard_normal_stats() {
        let n = 1_000_000;
        let samples = standard_rands(n);
        let (mean, variance) = calculate_stats(&samples);
        let std_dev = variance.sqrt();

        assert!(
            mean.abs() < 0.01,
            "The mean of the standard normal distribution should be close to 0, got {}",
            mean
        );
        assert!(
            (std_dev - 1.0).abs() < 0.01,
            "The standard deviation of the standard normal distribution should be close to 1, got {}",
            std_dev
        );
    }

    #[test]
    fn test_normal_stats() {
        let n = 1_000_000;
        let mu = 2.0;
        let sigma = 3.0;
        let samples = rands(mu, sigma, n).unwrap();
        let (mean, variance) = calculate_stats(&samples);
        let std_dev = variance.sqrt();

        assert!(
            (mean - mu).abs() < 0.05,
            "The mean of the normal distribution should be close to {}, got {}",
            mu,
            mean
        );
        assert!(
            (std_dev - sigma).abs() < 0.05,
            "The standard deviation of the normal distribution should be close to {}, got {}",
            sigma,
            std_dev
        );
    }
}
