//! Poisson distribution random number generation
//!
//! This module provides functions for generating random numbers from the Poisson distribution.

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
/// use diffusionx_core::random::poisson::rand;
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
/// use diffusionx_core::random::poisson::rands;
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

    #[test]
    fn test_rand() {
        let _random = rand(1.0).unwrap();
    }

    #[test]
    fn test_rands() {
        let randoms = rands(1.0, 10).unwrap();
        assert_eq!(randoms.len(), 10);
    }
}
