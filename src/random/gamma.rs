//! Gamma distribution random number generation

use crate::{XError, XResult};
use num_traits::float::Float;
use rand::{prelude::*, rng};
use rand_distr::{Exp1, Open01, StandardNormal};
use rayon::prelude::*;

/// Gamma distribution
#[derive(Debug, Clone)]
pub struct Gamma<T: Float + Send + Sync = f64> {
    /// shape parameter
    shape: T,
    /// scale parameter
    scale: T,
}

impl Default for Gamma {
    fn default() -> Self {
        Self {
            shape: 1.0,
            scale: 1.0,
        }
    }
}

impl<T: Float + Send + Sync> Gamma<T> {
    /// Create a new gamma distribution with a given shape and scale
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape parameter of the gamma distribution, must be greater than 0.
    /// * `scale` - The scale parameter of the gamma distribution, must be greater than 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::random::gamma::Gamma;
    ///
    /// let shape = 1.0;
    /// let scale = 2.0;
    /// let gamma = Gamma::new(shape, scale).unwrap();
    /// ```
    pub fn new(shape: T, scale: T) -> XResult<Self>
    where
        T: std::fmt::Display,
    {
        if shape <= T::zero() {
            return Err(XError::InvalidParameters(format!(
                "The shape parameter `shape` must be greater than 0, got {shape}"
            )));
        }
        if scale <= T::zero() {
            return Err(XError::InvalidParameters(format!(
                "The scale parameter `scale` must be greater than 0, got {scale}"
            )));
        }
        Ok(Self { shape, scale })
    }

    /// Get the shape parameter
    pub fn get_shape(&self) -> T {
        self.shape
    }

    /// Get the scale parameter
    pub fn get_scale(&self) -> T {
        self.scale
    }

    /// Generate a vector of gamma random numbers
    ///
    /// # Arguments
    ///
    /// * `n` - The number of random numbers to generate, must be greater than 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::random::gamma::Gamma;
    ///
    /// let gamma = Gamma::new(1.0, 1.0).unwrap();
    /// let randoms = gamma.samples(10).unwrap();
    /// ```
    pub fn samples(&self, n: usize) -> XResult<Vec<T>>
    where
        StandardNormal: Distribution<T>,
        Exp1: Distribution<T>,
        Open01: Distribution<T>,
    {
        rands(self.shape, self.scale, n)
    }
}

/// Generate a gamma random number
///
/// # Arguments
///
/// * `shape` - The shape parameter of the gamma distribution, must be greater than 0.
/// * `scale` - The scale parameter of the gamma distribution, must be greater than 0.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::gamma::rand;
///
/// let random = rand(1.0, 1.0).unwrap();
/// ```
pub fn rand<T: Float + Send + Sync>(shape: T, scale: T) -> XResult<T>
where
    StandardNormal: Distribution<T>,
    Exp1: Distribution<T>,
    Open01: Distribution<T>,
{
    let gamma = rand_distr::Gamma::new(shape, scale)
        .map_err(|e| XError::InvalidParameters(e.to_string()))?;
    Ok(rng().sample(gamma))
}

/// Generate a vector of gamma random numbers
///
/// # Arguments
///
/// * `shape` - The shape parameter of the gamma distribution, must be greater than 0.
/// * `scale` - The scale parameter of the gamma distribution, must be greater than 0.
/// * `n` - The number of random numbers to generate, must be greater than 0.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::gamma::rands;
///
/// let randoms = rands(1.0, 1.0, 10).unwrap();
/// ```
pub fn rands<T: Float + Send + Sync>(shape: T, scale: T, n: usize) -> XResult<Vec<T>>
where
    StandardNormal: Distribution<T>,
    Exp1: Distribution<T>,
    Open01: Distribution<T>,
{
    let gamma = rand_distr::Gamma::new(shape, scale)
        .map_err(|e| XError::InvalidParameters(e.to_string()))?;
    Ok((0..n)
        .into_par_iter()
        .map_init(rng, |r, _| r.sample(gamma))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::calculate_stats;

    #[test]
    fn test_rand() {
        let random = rand(1.0, 1.0).unwrap();
        assert!(random.is_finite());
    }

    #[test]
    fn test_rands() {
        let randoms = rands(1.0, 1.0, 10).unwrap();
        assert_eq!(randoms.len(), 10);
        assert!(randoms.iter().all(|r| r.is_finite()));
    }

    #[test]
    fn test_gamma_stats() {
        let n = 1_000_000;
        let shape = 1.0;
        let scale = 1.0;
        let samples = rands(shape, scale, n).unwrap();
        let (mean, variance) = calculate_stats(&samples);
        let std_dev = variance.sqrt();
        assert!(mean.is_finite());
        assert!(std_dev.is_finite());
    }
}
