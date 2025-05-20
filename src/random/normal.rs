//! Normal random number generation
//! For other stable distributions, see [crate::random::stable].
//!

use std::ops::{Add, Mul, Neg, Sub};

use crate::{XError, XResult};
use rand::{prelude::*, rng};
use rayon::prelude::*;

/// Normal distribution
pub struct Normal {
    /// mean
    mu: f64,
    /// standard deviation
    sigma: f64,
}

impl Default for Normal {
    fn default() -> Self {
        Self {
            mu: 0.0,
            sigma: 1.0,
        }
    }
}

impl Normal {
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
    pub fn new(mu: impl Into<f64>, sigma: impl Into<f64>) -> XResult<Self> {
        let mu = mu.into();
        let sigma = sigma.into();
        if sigma <= 0.0 {
            return Err(XError::InvalidParameters(format!(
                "The standard deviation `sigma` must be greater than 0, got {}",
                sigma
            )));
        }
        Ok(Self { mu, sigma })
    }

    /// Get the mean
    pub fn get_mu(&self) -> f64 {
        self.mu
    }

    /// Get the standard deviation
    pub fn get_sigma(&self) -> f64 {
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
    pub fn samples(&self, n: usize) -> XResult<Vec<f64>> {
        if self.sigma == 1.0 && self.mu == 0.0 {
            Ok(standard_rands(n))
        } else {
            rands(self.mu, self.sigma, n)
        }
    }
}

impl Neg for Normal {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            mu: -self.mu,
            sigma: self.sigma,
        }
    }
}

impl Add for Normal {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let sigma = (self.sigma * self.sigma + rhs.sigma * rhs.sigma).sqrt();
        Self {
            mu: self.mu + rhs.mu,
            sigma,
        }
    }
}

impl Sub for Normal {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<T: Into<f64>> Mul<T> for Normal {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();
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

impl Mul<Normal> for f32 {
    type Output = Normal;

    fn mul(self, rhs: Normal) -> Self::Output {
        let scale = self as f64;
        Self::Output {
            mu: scale * rhs.mu,
            sigma: scale.abs() * rhs.sigma,
        }
    }
}

impl Mul<Normal> for i32 {
    type Output = Normal;

    fn mul(self, rhs: Normal) -> Self::Output {
        let scale = self as f64;
        Self::Output {
            mu: scale * rhs.mu,
            sigma: scale.abs() * rhs.sigma,
        }
    }
}

impl Mul<Normal> for i64 {
    type Output = Normal;

    fn mul(self, rhs: Normal) -> Self::Output {
        let scale = self as f64;
        Self::Output {
            mu: scale * rhs.mu,
            sigma: scale.abs() * rhs.sigma,
        }
    }
}

impl<T: Into<f64>> Add<T> for Normal {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();
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

impl Add<Normal> for f32 {
    type Output = Normal;

    fn add(self, rhs: Normal) -> Self::Output {
        Self::Output {
            mu: self as f64 + rhs.mu,
            sigma: rhs.sigma,
        }
    }
}

impl Add<Normal> for i32 {
    type Output = Normal;

    fn add(self, rhs: Normal) -> Self::Output {
        Self::Output {
            mu: self as f64 + rhs.mu,
            sigma: rhs.sigma,
        }
    }
}

impl Add<Normal> for i64 {
    type Output = Normal;

    fn add(self, rhs: Normal) -> Self::Output {
        Self::Output {
            mu: self as f64 + rhs.mu,
            sigma: rhs.sigma,
        }
    }
}

impl<T: Into<f64>> Sub<T> for Normal {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();
        self + (-rhs)
    }
}

impl Sub<Normal> for f64 {
    type Output = Normal;

    fn sub(self, rhs: Normal) -> Self::Output {
        self + (-rhs)
    }
}

impl Sub<Normal> for f32 {
    type Output = Normal;

    fn sub(self, rhs: Normal) -> Self::Output {
        self + (-rhs)
    }
}

impl Sub<Normal> for i32 {
    type Output = Normal;

    fn sub(self, rhs: Normal) -> Self::Output {
        self + (-rhs)
    }
}

impl Sub<Normal> for i64 {
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
/// let random = standard_rand();
/// ```
pub fn standard_rand() -> f64 {
    rng().sample(rand_distr::StandardNormal)
}

/// Generate a vector of standard normal random numbers
///
/// # Example
///
/// ```rust
/// use diffusionx::random::normal::standard_rands;
///
/// let randoms = standard_rands(10);
/// ```
pub fn standard_rands(n: usize) -> Vec<f64> {
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
pub fn rand(mean: impl Into<f64>, std_dev: impl Into<f64>) -> XResult<f64> {
    let mean = mean.into();
    let std_dev = std_dev.into();
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
pub fn rands(mean: impl Into<f64>, std_dev: impl Into<f64>, n: usize) -> XResult<Vec<f64>> {
    let mean = mean.into();
    let std_dev = std_dev.into();
    if std_dev <= 0.0 {
        return Err(XError::InvalidParameters(format!(
            "The standard deviation `std_dev` must be greater than 0, got {}",
            std_dev
        )));
    }
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
