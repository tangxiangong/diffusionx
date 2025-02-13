//! Normal random number generation
//!
//! This module provides functions for generating normal random numbers.

use crate::XResult;
use rand::{prelude::*, rng};
use rand_distr::{Normal, StandardNormal};
use rayon::prelude::*;

/// Generate a standard normal random number
///
/// This function generates a standard normal random number using the `StandardNormal` distribution.
///
/// # Returns
///
/// A `f64` value representing the generated random number.
///
/// # Example
///
/// ```rust
/// use diffusionx_core::random::normal::standard_rand;
/// let random = standard_rand();
/// ```
pub fn standard_rand() -> f64 {
    rng().sample(StandardNormal)
}

/// Generate a vector of standard normal random numbers
///
/// This function generates a vector of standard normal random numbers using the `StandardNormal` distribution.
///
/// # Returns
///
/// A vector of `f64` values representing the generated random numbers.
///
/// # Example
///
/// ```rust
/// use diffusionx_core::random::normal::standard_rands;
/// let randoms = standard_rands(10);
/// ```
pub fn standard_rands(n: usize) -> Vec<f64> {
    let dist = StandardNormal;
    (0..n).into_par_iter().map_init(rng, |r, _|r.sample(dist)).collect()
}

/// Generate a normal random number
///
/// This function generates a normal random number using the `Normal` distribution.
///
/// # Returns
///
/// A `f64` value representing the generated random number.
///
/// # Example
///
/// ```rust
/// use diffusionx_core::random::normal::rand;
/// let random = rand(0.0, 1.0);
/// ```
pub fn rand(mean: impl Into<f64>, std_dev: impl Into<f64>) -> XResult<f64> {
    let mean = mean.into();
    let std_dev = std_dev.into();
    let normal = Normal::new(mean, std_dev)?;
    Ok(rng().sample(normal))
}

/// Generate a vector of normal random numbers
///
/// This function generates a vector of normal random numbers using the `Normal` distribution.
///
/// # Returns
///
/// A vector of `f64` values representing the generated random numbers.
///
/// # Example
///
/// ```rust
/// use diffusionx_core::random::normal::rands;
/// let randoms = rands(0.0, 1.0, 10);
/// ```
pub fn rands(mean: impl Into<f64>, std_dev: impl Into<f64>, n: usize) -> XResult<Vec<f64>> {
    let mean = mean.into();
    let std_dev = std_dev.into();
    let normal = Normal::new(mean, std_dev)?;
    Ok((0..n)
        .into_par_iter()
        .map_init(rng, |r, _| r.sample(normal))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
