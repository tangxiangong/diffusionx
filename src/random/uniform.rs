//! Uniform random number generation

use crate::{FloatExt, XError, XResult, random::PAR_THRESHOLD};
use rand::{
    distr::{
        StandardUniform,
        uniform::{SampleUniform, Uniform},
    },
    prelude::*,
};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::ops::{Range, RangeInclusive};

/// Generate a standard uniform random number
///
/// # Example
///
/// ```rust
/// use diffusionx::random::uniform::standard_rand;
///
/// let random = standard_rand();
/// assert!((0.0..1.0).contains(&random));
/// ```
pub fn standard_rand<T: FloatExt>() -> T
where
    StandardUniform: Distribution<T>,
{
    let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
    rng.sample(StandardUniform)
}

/// Generate a vector of standard uniform random numbers
///
/// # Example
///
/// ```rust
/// use diffusionx::random::uniform::standard_rands;
///
/// let randoms = standard_rands(10);
/// assert_eq!(randoms.len(), 10);
/// assert!(randoms.iter().all(|x| (0.0..1.0).contains(x)));
/// ```
pub fn standard_rands<T: FloatExt>(n: usize) -> Vec<T>
where
    StandardUniform: Distribution<T>,
{
    let dist = StandardUniform;
    if n <= PAR_THRESHOLD {
        let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
        (0..n).map(|_| rng.sample(dist)).collect()
    } else {
        (0..n)
            .into_par_iter()
            .map_init(
                || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                |r, _| r.sample(dist),
            )
            .collect()
    }
}

/// Generate a random number from a range
///
/// # Example
///
/// ```rust
/// use diffusionx::random::uniform::range_rand;
///
/// let random = range_rand(0..10).unwrap();
/// assert!((0..10).contains(&random));
/// ```
pub fn range_rand<T>(range: Range<T>) -> XResult<T>
where
    T: SampleUniform,
{
    let uniform = Uniform::new(range.start, range.end)?;
    let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
    Ok(rng.sample(uniform))
}

/// Generate a vector of random numbers from a range
///
/// # Example
///
/// ```rust
/// use diffusionx::random::uniform::range_rands;
///
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
    if n <= PAR_THRESHOLD {
        let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
        Ok((0..n).map(|_| rng.sample(uniform)).collect())
    } else {
        Ok((0..n)
            .into_par_iter()
            .map_init(
                || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                |r, _| r.sample(uniform),
            )
            .collect())
    }
}

/// Generate a random number from an inclusive range
///
/// # Example
///
/// ```rust
/// use diffusionx::random::uniform::inclusive_range_rand;
///
/// let random = inclusive_range_rand(0..=10).unwrap();
/// assert!((0..=10).contains(&random));
/// ```
pub fn inclusive_range_rand<T>(range: RangeInclusive<T>) -> XResult<T>
where
    T: SampleUniform,
{
    let uniform = Uniform::new_inclusive(range.start(), range.end())?;
    let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
    Ok(rng.sample(uniform))
}

/// Generate a vector of random numbers from an inclusive range
///
/// # Example
///
/// ```rust
/// use diffusionx::random::uniform::inclusive_range_rands;
///
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
    if n <= PAR_THRESHOLD {
        let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
        Ok((0..n).map(|_| rng.sample(uniform)).collect())
    } else {
        Ok((0..n)
            .into_par_iter()
            .map_init(
                || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                |r, _| r.sample(uniform),
            )
            .collect())
    }
}

/// Generate a boolean random number
///
/// # Example
///
/// ```rust
/// use diffusionx::random::uniform::bool_rand;
///
/// let random = bool_rand(0.5).unwrap();
/// println!("random: {}", random);
/// ```
pub fn bool_rand(p: f64) -> XResult<bool> {
    if !(0.0..=1.0).contains(&p) {
        return Err(XError::BoolSampleError);
    }
    let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
    let result = rng.random_bool(p);
    Ok(result)
}

/// Generate a vector of boolean random numbers
///
/// # Example
///
/// ```rust
/// use diffusionx::random::uniform::bool_rands;
///
/// let randoms = bool_rands(0.5, 10).unwrap();
/// println!("randoms: {:?}", randoms);
/// ```
pub fn bool_rands(p: f64, n: usize) -> XResult<Vec<bool>> {
    if !(0.0..=1.0).contains(&p) {
        return Err(XError::BoolSampleError);
    }
    if n <= PAR_THRESHOLD {
        let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
        Ok((0..n).map(|_| rng.random_bool(p)).collect())
    } else {
        Ok((0..n)
            .into_par_iter()
            .map_init(
                || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                |r, _| r.random_bool(p),
            )
            .collect())
    }
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
        assert_eq!(randoms.len() as usize, n);
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
        assert_eq!(randoms.len() as usize, n);
        assert!(randoms.iter().all(|x| (0..=10).contains(x)));
    }

    #[test]
    fn test_standard_uniform_stats() {
        let n = 1_000_000;
        let samples = standard_rands(n);
        let (mean, variance) = calculate_stats(&samples);

        assert!(
            (mean - 0.5).abs() < 0.01,
            "The mean of the standard uniform distribution should be close to 0.5, but got {mean}"
        );

        let expected_variance = 1.0 / 12.0;
        assert!(
            (variance - expected_variance).abs() < 0.01,
            "The variance of the standard uniform distribution should be close to {expected_variance}, but got {variance}"
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
            "The mean of the uniform distribution should be close to {expected_mean}, but got {mean}"
        );
        assert!(
            (variance - expected_variance).abs() < 0.01,
            "The variance of the uniform distribution should be close to {expected_variance}, but got {variance}"
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
            "The proportion of True in the boolean random numbers should be close to {p}, but got {mean}"
        );
    }
}
