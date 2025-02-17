//! Uniform random number generation

use crate::{XError, XResult};
use rand::{
    distr::{
        StandardUniform,
        uniform::{SampleUniform, Uniform},
    },
    prelude::*,
    rng,
};
use rayon::prelude::*;
use std::ops::{Range, RangeInclusive};

/// Generate a standard uniform random number
///
/// This function generates a standard uniform random number using the `StandardUniform` distribution.
///
/// # Returns
///
/// A `f64` value representing the generated random number.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::uniform::standard_rand;
/// let random = standard_rand();
/// assert!((0.0..1.0).contains(&random));
/// ```
pub fn standard_rand() -> f64 {
    rng().sample(StandardUniform)
}

/// Generate a vector of standard uniform random numbers
///
/// This function generates a vector of standard uniform random numbers using the `StandardUniform` distribution.
///
/// # Returns
///
/// A vector of `f64` values representing the generated random numbers.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::uniform::standard_rands;
/// let randoms = standard_rands(10);
/// assert_eq!(randoms.len(), 10);
/// assert!(randoms.iter().all(|x| (0.0..1.0).contains(x)));
/// ```
pub fn standard_rands(n: usize) -> Vec<f64> {
    let dist = StandardUniform;
    (0..n)
        .into_par_iter()
        .map_init(rng, |r, _| r.sample(dist))
        .collect()
}

/// Generate a random number from a range
///
/// This function generates a random number from a range using the `Uniform` distribution.
///
/// # Returns
///
/// A `T` value representing the generated random number.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::uniform::range_rand;
/// let random = range_rand(0..10).unwrap();
/// assert!((0..10).contains(&random));
/// ```
pub fn range_rand<T>(range: Range<T>) -> XResult<T>
where
    T: SampleUniform,
{
    let uniform = Uniform::new(range.start, range.end)?;
    Ok(rng().sample(uniform))
}

/// Generate a vector of random numbers from a range
///
/// This function generates a vector of random numbers from a range using the `Uniform` distribution.
///
/// # Returns
///
/// A vector of `T` values representing the generated random numbers.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::uniform::range_rands;
/// let randoms = range_rands(0..10, 10).unwrap();
/// assert_eq!(randoms.len(), 10);
/// assert!(randoms.iter().all(|x| (0..10).contains(x)));
/// ```
pub fn range_rands<T>(range: Range<T>, n: usize) -> XResult<Vec<T>>
where
    T: SampleUniform + Send + Sync,
    Uniform<T>: Copy,
    <T as SampleUniform>::Sampler: Send + Sync,
{
    let uniform = Uniform::new(range.start, range.end)?;
    let result = (0..n)
        .into_par_iter()
        .map_init(rng, |r, _| r.sample(uniform))
        .collect();
    Ok(result)
}

/// Generate a random number from an inclusive range
///
/// This function generates a random number from an inclusive range using the `Uniform` distribution.
///
/// # Returns
///
/// A `T` value representing the generated random number.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::uniform::inclusive_range_rand;
/// let random = inclusive_range_rand(0..=10).unwrap();
/// assert!((0..=10).contains(&random));
/// ```
pub fn inclusive_range_rand<T>(range: RangeInclusive<T>) -> XResult<T>
where
    T: SampleUniform,
{
    let uniform = Uniform::new_inclusive(range.start(), range.end())?;
    Ok(rng().sample(uniform))
}

/// Generate a vector of random numbers from an inclusive range
///
/// This function generates a vector of random numbers from an inclusive range using the `Uniform` distribution.
///
/// # Returns
///
/// A vector of `T` values representing the generated random numbers.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::uniform::inclusive_range_rands;
/// let randoms = inclusive_range_rands(0..=10, 10).unwrap();
/// assert_eq!(randoms.len(), 10);
/// assert!(randoms.iter().all(|x| (0..=10).contains(x)));
/// ```
pub fn inclusive_range_rands<T>(range: RangeInclusive<T>, n: usize) -> XResult<Vec<T>>
where
    T: SampleUniform + Send + Sync,
    Uniform<T>: Copy,
    <T as SampleUniform>::Sampler: Send + Sync,
{
    let uniform = Uniform::new_inclusive(range.start(), range.end())?;
    let result = (0..n)
        .into_par_iter()
        .map_init(rng, |r, _| r.sample(uniform))
        .collect();
    Ok(result)
}

/// Generate a boolean random number
///
/// This function generates a boolean random number with a given probability.
///
/// # Returns
///
/// A `bool` value representing the generated random number.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::uniform::bool_rand;
/// let random = bool_rand(0.5).unwrap();
/// println!("random: {}", random);
/// ```
pub fn bool_rand(p: f64) -> XResult<bool> {
    if !(0.0..=1.0).contains(&p) {
        return Err(XError::BoolSampleError);
    }
    let result = rng().random_bool(p);
    Ok(result)
}

/// Generate a vector of boolean random numbers
///
/// This function generates a vector of boolean random numbers with a given probability.
///
/// # Returns
///
/// A vector of `bool` values representing the generated random numbers.    
///
/// # Example
///
/// ```rust
/// use diffusionx::random::uniform::bool_rands;
/// let randoms = bool_rands(0.5, 10).unwrap();
/// println!("randoms: {:?}", randoms);
/// ```
pub fn bool_rands(p: f64, n: usize) -> XResult<Vec<bool>> {
    if !(0.0..=1.0).contains(&p) {
        return Err(XError::BoolSampleError);
    }
    let result = (0..n)
        .into_par_iter()
        .map_init(rng, |r, _| r.random_bool(p))
        .collect();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{calculate_bool_mean, calculate_stats};

    #[test]
    fn test_unit_random() {
        let random = standard_rand();
        assert!((0.0..1.0).contains(&random));
    }

    #[test]
    fn test_unit_randoms() {
        let n = 1000000;
        let randoms = standard_rands(n);
        assert_eq!(randoms.len(), n);
        assert!(randoms.iter().all(|x| (0.0..1.0).contains(x)));
    }

    #[test]
    fn test_range_random() {
        let random = range_rand(0..10).unwrap();
        assert!((0..10).contains(&random));
    }

    #[test]
    fn test_range_randoms() {
        let n = 1000000;
        let randoms = range_rands(0..10, n).unwrap();
        assert_eq!(randoms.len(), n);
        assert!(randoms.iter().all(|x| (0..10).contains(x)));
    }

    #[test]
    fn test_inclusive_range_random() {
        let random = inclusive_range_rand(0..=10).unwrap();
        assert!((0..=10).contains(&random));
    }

    #[test]
    fn test_inclusive_range_randoms() {
        let n = 1000000;
        let randoms = inclusive_range_rands(0..=10, n).unwrap();
        assert_eq!(randoms.len(), n);
        assert!(randoms.iter().all(|x| (0..=10).contains(x)));
    }

    #[test]
    fn test_standard_uniform_stats() {
        let n = 1_000_000;
        let samples = standard_rands(n);
        let (mean, variance) = calculate_stats(&samples);

        assert!(
            (mean - 0.5).abs() < 0.01,
            "标准均匀分布的均值应接近0.5，实际为{}",
            mean
        );

        let expected_variance = 1.0 / 12.0;
        assert!(
            (variance - expected_variance).abs() < 0.01,
            "标准均匀分布的方差应接近{}，实际为{}",
            expected_variance,
            variance
        );
    }

    #[test]
    fn test_range_uniform_stats() {
        let n = 1_000_000;
        let a = -2.0;
        let b = 3.0;
        let samples = range_rands(a..b, n).unwrap();
        let (mean, variance) = calculate_stats(&samples);

        let expected_mean = (a + b) / 2.0;
        let expected_variance = (b - a).powi(2) / 12.0;

        assert!(
            (mean - expected_mean).abs() < 0.01,
            "均匀分布的均值应接近{}，实际为{}",
            expected_mean,
            mean
        );
        assert!(
            (variance - expected_variance).abs() < 0.01,
            "均匀分布的方差应接近{}，实际为{}",
            expected_variance,
            variance
        );
    }

    #[test]
    fn test_bool_rand_stats() {
        let n = 1_000_000;
        let p = 0.7;
        let samples = bool_rands(p, n).unwrap();
        let mean = calculate_bool_mean(&samples);

        assert!(
            (mean - p).abs() < 0.01,
            "布尔随机数中True的比例应接近{}，实际为{}",
            p,
            mean
        );
    }
}
