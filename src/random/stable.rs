//! Generate Lévy stable distribution random numbers.
//!
//! For the Gaussian distribution, see [crate::random::normal].
//!
//! Stable laws – also called $\alpha$-stable, stable Paretian or Lévy stable – were
//! introduced by Lévy (1925) during his investigations of the behavior of sums of
//! independent random variables. A sum of two independent random variables having
//! an $\alpha$-stable distribution with index $\alpha$ is again $\alpha$-stable with the
//! same index $\alpha$. This invariance property, however, does not hold for
//! different $\alpha$'s.
//!
//! The $\alpha$-stable distribution requires four parameters for complete description:
//! - an index of stability $\alpha$ in (0, 2],
//! - a skewness parameter $\beta$ in [-1, 1],
//! - a positive scale parameter $\sigma$, and
//! - a real location parameter $\mu$.
//!
//! The tail exponent $\alpha$ determines the rate at which the tails of the distribution taper off.
//! When $\alpha = 2$, the Gaussian distribution results. When $\alpha < 2$, the variance
//! is infinite and the tails are asymptotically equivalent to a Pareto law, i.e. they
//! decay as a power law.
//!
//! When the skewness parameter $\beta$ is positive, the distribution is skewed to the right,
//! i.e. the right tail is thicker. When it is negative, it is skewed to the left.
//! When $\beta = 0$, the distribution is symmetric about $\mu$. As $\alpha$ approaches 2,
//! $\beta$ loses its effect and the distribution approaches the Gaussian
//! distribution regardless of $\beta$. The last two parameters, $\sigma$ and $\mu$, are the usual
//! scale and location parameters, i.e. $\sigma$ determines the width and $\mu$ the shift of
//! the mode (the peak) of the density. For $\sigma = 1$ and $\mu = 0$ the distribution is called
//! the standard alpha-stable distribution.
//!
//! # References
//!
//! [Borak, Szymon; Härdle, Wolfgang Karl; Weron, Rafał (2005) : Stable distributions,
//! SFB 649 Discussion Paper, No. 2005-008, Humboldt University of Berlin, Collaborative Research
//! Center 649 - Economic Risk, Berlin](https://hdl.handle.net/10419/25027)

use crate::{StableError, XResult};
use rand::{Rng, prelude::*};
use rand_distr::Exp1;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::{
    f64::consts::{FRAC_PI_2, PI},
    ops::{Add, Mul},
};

/// Precomputed constants for stable distribution sampling
#[derive(Debug, Clone, Copy)]
struct StableConstants {
    /// 1.0 / alpha
    inv_alpha: f64,
    /// (1.0 - alpha) / alpha
    one_minus_alpha_div_alpha: f64,
    /// atan(beta * tan(alpha * PI/2)) / alpha
    b: f64,
    /// (1.0 + (beta * tan(alpha * PI/2))^2)^(1/(2*alpha))
    s: f64,
}

impl StableConstants {
    #[inline]
    fn new(alpha: f64, beta: f64) -> Self {
        let inv_alpha = 1.0 / alpha;
        let one_minus_alpha_div_alpha = (1.0 - alpha) * inv_alpha;
        let tmp = beta * (alpha * FRAC_PI_2).tan();
        let b = tmp.atan() * inv_alpha;
        let s = (1.0 + tmp * tmp).powf(0.5 * inv_alpha);
        Self {
            inv_alpha,
            b,
            s,
            one_minus_alpha_div_alpha,
        }
    }
}

/// Standard Lévy stable distribution
///
/// i.e., with scale parameter 1 and location parameter 0
#[derive(Debug, Clone)]
pub struct StandardStable {
    /// Index of stability
    alpha: f64,
    /// Skewness parameter
    beta: f64,
}

impl StandardStable {
    /// Create a new standard Lévy stable distribution
    ///
    /// # Arguments
    ///
    /// * `alpha` - The index of stability, must be in the range (0, 2].
    /// * `beta` - The skewness parameter, must be in the range [-1, 1].
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::random::stable::StandardStable;
    ///
    /// let stable = StandardStable::new(0.7, 1.0).unwrap();
    /// ```
    pub fn new(alpha: impl Into<f64>, beta: impl Into<f64>) -> XResult<Self> {
        let alpha: f64 = alpha.into();
        let beta: f64 = beta.into();
        if alpha <= 0.0 || alpha > 2.0 || alpha.is_nan() {
            return Err(StableError::InvalidIndex.into());
        }
        if !(-1.0..=1.0).contains(&beta) {
            return Err(StableError::InvalidSkewness.into());
        }
        Ok(Self { alpha, beta })
    }

    /// Get the index of stability
    pub fn get_index(&self) -> f64 {
        self.alpha
    }

    /// Get the skewness parameter
    pub fn get_skewness(&self) -> f64 {
        self.beta
    }

    /// Sample from the standard Lévy stable distribution
    ///
    /// # Arguments
    ///
    /// * `n` - The number of samples to generate, must be greater than 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::random::stable::StandardStable;
    ///
    /// let stable = StandardStable::new(0.7, 1.0).unwrap();
    /// let samples = stable.samples(10).unwrap();
    /// println!("samples: {:?}", samples);
    /// ```
    pub fn samples(&self, n: usize) -> XResult<Vec<f64>> {
        standard_rands(self.alpha, self.beta, n)
    }
}

/// Sample standard stable random number when alpha is not 1
pub(crate) fn sample_standard_alpha<R: Rng + ?Sized>(alpha: f64, beta: f64, rng: &mut R) -> f64 {
    let constants = StableConstants::new(alpha, beta);
    sample_standard_alpha_with_constants(&constants, alpha, rng)
}

/// Sample standard stable random number with precomputed constants
#[inline]
fn sample_standard_alpha_with_constants<R: Rng + ?Sized>(
    c: &StableConstants,
    alpha: f64,
    rng: &mut R,
) -> f64 {
    let v = rng.random_range(-FRAC_PI_2..FRAC_PI_2);
    let w: f64 = rng.sample(Exp1);
    let v_plus_b = v + c.b;
    let cos_v = v.cos();
    // alpha * sin(v + b) / cos(v)^(1/alpha)
    let c1 = alpha * v_plus_b.sin() / cos_v.powf(c.inv_alpha);
    // ((cos(v - alpha*(v+b)) / w))^((1-alpha)/alpha)
    let c2 = ((v - alpha * v_plus_b).cos() / w).powf(c.one_minus_alpha_div_alpha);
    c.s * c1 * c2
}

/// Sample standard stable random number when alpha is 1
#[inline]
pub(crate) fn sample_standard_alpha_one<R: Rng + ?Sized>(
    _alpha: f64,
    beta: f64,
    rng: &mut R,
) -> f64 {
    let v = rng.random_range(-FRAC_PI_2..FRAC_PI_2);
    let w: f64 = rng.sample(Exp1);
    let half_pi_plus_beta_v = FRAC_PI_2 + beta * v;
    let c1 = half_pi_plus_beta_v * v.tan();
    let c2 = ((FRAC_PI_2 * w * v.cos()) / half_pi_plus_beta_v).ln() * beta;
    (c1 - c2) * std::f64::consts::FRAC_2_PI
}

/// Sample from the standard Lévy stable distribution
impl Distribution<f64> for StandardStable {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        if (self.alpha - 1.0).abs() > 1e-10 {
            sample_standard_alpha(self.alpha, self.beta, rng)
        } else {
            sample_standard_alpha_one(self.alpha, self.beta, rng)
        }
    }
}

/// Lévy stable distribution
#[derive(Debug, Clone, Copy)]
pub struct Stable {
    /// Index of stability
    alpha: f64,
    /// Skewness parameter
    beta: f64,
    /// Scale parameter
    sigma: f64,
    /// Location parameter
    mu: f64,
}

/// Convert a standard Lévy stable distribution to a Lévy stable distribution
impl From<&Stable> for StandardStable {
    fn from(stable: &Stable) -> Self {
        StandardStable::new(stable.alpha, stable.beta).unwrap()
    }
}

impl Stable {
    /// Create a new Lévy stable distribution
    ///
    /// # Arguments
    ///
    /// * `alpha` - The index of stability, must be in the range (0, 2].
    /// * `beta` - The skewness parameter, must be in the range [-1, 1].
    /// * `sigma` - The scale parameter, must be greater than 0.
    /// * `mu` - The location parameter.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::random::stable::Stable;
    ///
    /// let stable = Stable::new(0.7, 1.0, 1.0, 0.0).unwrap();
    /// ```
    pub fn new(
        alpha: impl Into<f64>,
        beta: impl Into<f64>,
        sigma: impl Into<f64>,
        mu: impl Into<f64>,
    ) -> XResult<Self> {
        let alpha: f64 = alpha.into();
        let beta: f64 = beta.into();
        let sigma: f64 = sigma.into();
        let mu: f64 = mu.into();
        if alpha <= 0.0 || alpha > 2.0 || alpha.is_nan() {
            return Err(StableError::InvalidIndex.into());
        }
        if !(-1.0..=1.0).contains(&beta) {
            return Err(StableError::InvalidSkewness.into());
        }
        if sigma <= 0.0 || sigma.is_nan() {
            return Err(StableError::InvalidScale.into());
        }
        if mu.is_nan() {
            return Err(StableError::InvalidLocation.into());
        }
        Ok(Self {
            alpha,
            beta,
            sigma,
            mu,
        })
    }

    /// Get the index of stability
    pub fn get_index(&self) -> f64 {
        self.alpha
    }

    /// Get the skewness parameter
    pub fn get_skewness(&self) -> f64 {
        self.beta
    }

    /// Get the scale parameter
    pub fn get_scale(&self) -> f64 {
        self.sigma
    }

    /// Get the location parameter
    pub fn get_location(&self) -> f64 {
        self.mu
    }

    /// Sample from the Lévy stable distribution
    ///
    /// # Arguments
    ///
    /// * `n` - The number of samples to generate, must be greater than 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::random::stable::Stable;
    ///
    /// let stable = Stable::new(0.7, 1.0, 1.0, 0.0).unwrap();
    /// let samples = stable.samples(10).unwrap();
    /// println!("samples: {:?}", samples);
    /// ```
    pub fn samples(&self, n: usize) -> XResult<Vec<f64>> {
        rands(self.alpha, self.beta, self.sigma, self.mu, n)
    }
}

/// Sample from the Lévy stable distribution
impl Distribution<f64> for Stable {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let standard = StandardStable::from(self);
        let r = rng.sample(standard);
        if self.alpha != 1.0 {
            self.sigma * r + self.mu
        } else {
            self.sigma * r + self.mu + 2.0 * self.beta * self.sigma * self.sigma.ln() / PI
        }
    }
}

/// Standard skew Lévy stable distribution
#[derive(Debug, Clone, Copy)]
pub struct StandardSkewStable(pub f64);

impl StandardSkewStable {
    /// Create a new standard skew Lévy stable distribution
    ///
    /// # Arguments
    ///
    /// * `alpha` - The index of stability, must be in the range (0, 1).
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::random::stable::StandardSkewStable;
    ///
    /// let stable = StandardSkewStable::new(0.7).unwrap();
    /// ```
    pub fn new(alpha: impl Into<f64>) -> XResult<Self> {
        let alpha: f64 = alpha.into();
        if alpha <= 0.0 || alpha >= 1.0 || alpha.is_nan() {
            return Err(StableError::InvalidSkewIndex.into());
        }
        Ok(Self(alpha))
    }

    /// Get the index of stability
    pub fn get_index(&self) -> f64 {
        self.0
    }

    /// Sample from the standard skew Lévy stable distribution
    ///
    /// # Arguments
    ///
    /// * `n` - The number of samples to generate, must be greater than 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::random::stable::StandardSkewStable;
    ///
    /// let stable = StandardSkewStable::new(0.7).unwrap();
    /// let samples = stable.samples(10).unwrap();
    /// println!("samples: {:?}", samples);
    /// ```
    pub fn samples(&self, n: usize) -> XResult<Vec<f64>> {
        skew_rands(self.0, n)
    }
}

/// Sample standard skew stable random number
///
/// # Panic
///
/// if the skew index is invalid
impl Distribution<f64> for StandardSkewStable {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let alpha = self.0;
        if alpha <= 0.0 || alpha >= 1.0 || alpha.is_nan() {
            panic!("Invalid skew index");
        }
        sample_standard_alpha(self.0, 1.0, rng)
    }
}

/// Symmetric Lévy stable distribution
#[derive(Debug, Clone, Copy)]
pub struct SymmetricStandardStable(pub f64);

impl SymmetricStandardStable {
    /// Create a new symmetric standard Lévy stable distribution
    ///
    /// # Arguments
    ///
    /// * `alpha` - The index of stability, must be in the range (0, 2].
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::random::stable::SymmetricStandardStable;
    ///
    /// let stable = SymmetricStandardStable::new(0.7).unwrap();
    /// ```
    pub fn new(alpha: impl Into<f64>) -> XResult<Self> {
        let alpha: f64 = alpha.into();
        if alpha <= 0.0 || alpha >= 2.0 || alpha.is_nan() {
            return Err(StableError::InvalidSkewIndex.into());
        }
        Ok(Self(alpha))
    }

    /// Get the index of stability
    pub fn get_index(&self) -> f64 {
        self.0
    }

    /// Sample the symmetric standard Lévy stable distribution random numbers
    ///
    /// # Arguments
    ///
    /// * `n` - The number of samples to generate, must be greater than 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::random::stable::SymmetricStandardStable;
    ///
    /// let stable = SymmetricStandardStable::new(0.7).unwrap();
    /// let samples = stable.samples(10).unwrap();
    /// println!("samples: {:?}", samples);
    /// ```
    pub fn samples(&self, n: usize) -> XResult<Vec<f64>> {
        sym_standard_rands(self.0, n)
    }
}

/// Sample symmetric standard stable random number
///
/// # Panic
///
/// if the stability index is invalid
impl Distribution<f64> for SymmetricStandardStable {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let alpha = self.0;
        if alpha <= 0.0 || alpha > 2.0 || alpha.is_nan() {
            panic!("Invalid stability index");
        }
        sample_standard_alpha(self.0, 0.0, rng)
    }
}

/// Sample the standard Lévy stable distribution random number
///
/// # Arguments
///
/// * `alpha` - The index of stability, must be in the range (0, 2].
/// * `beta` - The skewness parameter, must be in the range [-1, 1].
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::standard_rand;
///
/// let alpha = 0.7;
/// let beta = 1.0;
/// let r = standard_rand(alpha, beta).unwrap();
/// println!("r: {}", r);
/// ```
pub fn standard_rand(alpha: impl Into<f64>, beta: impl Into<f64>) -> XResult<f64> {
    let standard = StandardStable::new(alpha, beta)?;
    let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
    Ok(rng.sample(standard))
}

/// Sample the standard Lévy stable distribution random numbers
///
/// # Arguments
///
/// * `alpha` - The index of stability, must be in the range (0, 2].
/// * `beta` - The skewness parameter, must be in the range [-1, 1].
/// * `n` - The number of samples to generate, must be greater than 0.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::standard_rands;
///
/// let alpha = 0.7;
/// let beta = 1.0;
/// let n = 10;
/// let r = standard_rands(alpha, beta, n).unwrap();
/// println!("r: {:?}", r);
/// ```
pub fn standard_rands(alpha: impl Into<f64>, beta: impl Into<f64>, n: usize) -> XResult<Vec<f64>> {
    let alpha: f64 = alpha.into();
    let beta: f64 = beta.into();
    if alpha <= 0.0 || alpha > 2.0 || alpha.is_nan() {
        return Err(StableError::InvalidIndex.into());
    }
    if !(-1.0..=1.0).contains(&beta) {
        return Err(StableError::InvalidSkewness.into());
    }
    if (alpha - 1.0).abs() < 1e-10 {
        Ok((0..n)
            .into_par_iter()
            .map_init(
                || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                |r, _| sample_standard_alpha_one(alpha, beta, r),
            )
            .collect())
    } else {
        let constants = StableConstants::new(alpha, beta);
        Ok((0..n)
            .into_par_iter()
            .map_init(
                || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                |r, _| sample_standard_alpha_with_constants(&constants, alpha, r),
            )
            .collect())
    }
}

/// Sample the Lévy stable distribution random number
///
/// # Arguments
///
/// * `alpha` - The index of stability, must be in the range (0, 2].
/// * `beta` - The skewness parameter, must be in the range [-1, 1].
/// * `sigma` - The scale parameter, must be greater than 0.
/// * `mu` - The location parameter.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::rand;
///
/// let alpha = 0.7;
/// let beta = 1.0;
/// let sigma = 1.0;
/// let mu = 0.0;
/// let r = rand(alpha, beta, sigma, mu).unwrap();
/// println!("r: {}", r);
/// ```
pub fn rand(
    alpha: impl Into<f64>,
    beta: impl Into<f64>,
    sigma: impl Into<f64>,
    mu: impl Into<f64>,
) -> XResult<f64> {
    let levy = Stable::new(alpha, beta, sigma, mu)?;
    let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
    Ok(rng.sample(levy))
}

/// Sample the Lévy stable distribution random numbers
///
/// # Arguments
///
/// * `alpha` - The index of stability, must be in the range (0, 2].
/// * `beta` - The skewness parameter, must be in the range [-1, 1].
/// * `sigma` - The scale parameter, must be greater than 0.
/// * `mu` - The location parameter.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::rands;
///
/// let alpha = 0.7;
/// let beta = 1.0;
/// let sigma = 1.0;
/// let mu = 0.0;
/// let n = 10;
/// let r = rands(alpha, beta, sigma, mu, n).unwrap();
/// assert_eq!(r.len(), n);
/// ```
pub fn rands(
    alpha: impl Into<f64>,
    beta: impl Into<f64>,
    sigma: impl Into<f64>,
    mu: impl Into<f64>,
    n: usize,
) -> XResult<Vec<f64>> {
    let alpha: f64 = alpha.into();
    let beta: f64 = beta.into();
    let sigma: f64 = sigma.into();
    let mu: f64 = mu.into();
    if alpha <= 0.0 || alpha > 2.0 || alpha.is_nan() {
        return Err(StableError::InvalidIndex.into());
    }
    if !(-1.0..=1.0).contains(&beta) {
        return Err(StableError::InvalidSkewness.into());
    }
    if sigma <= 0.0 || sigma.is_nan() {
        return Err(StableError::InvalidScale.into());
    }
    if mu.is_nan() {
        return Err(StableError::InvalidLocation.into());
    }
    if (alpha - 1.0).abs() < 1e-10 {
        let correction = 2.0 * beta * sigma * sigma.ln() / PI;
        Ok((0..n)
            .into_par_iter()
            .map_init(
                || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                |r, _| {
                    let std_sample = sample_standard_alpha_one(alpha, beta, r);
                    sigma * std_sample + mu + correction
                },
            )
            .collect())
    } else {
        let constants = StableConstants::new(alpha, beta);
        Ok((0..n)
            .into_par_iter()
            .map_init(
                || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                |r, _| {
                    let std_sample = sample_standard_alpha_with_constants(&constants, alpha, r);
                    sigma * std_sample + mu
                },
            )
            .collect())
    }
}

/// Sample the standard skew Lévy stable distribution random number
///
/// # Arguments
///
/// * `alpha` - The index of stability, must be in the range (0, 1).
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::skew_rand;
///
/// let alpha = 0.7;
/// let r = skew_rand(alpha).unwrap();
/// println!("r: {}", r);
/// ```
pub fn skew_rand(alpha: impl Into<f64>) -> XResult<f64> {
    let skew = StandardSkewStable::new(alpha)?;
    let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
    Ok(rng.sample(skew))
}

/// Sample the standard skew Lévy stable distribution random numbers
///
/// # Arguments
///
/// * `alpha` - The index of stability, must be in the range (0, 1).
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::skew_rands;
///
/// let alpha = 0.7;
/// let n = 10;
/// let r = skew_rands(alpha, n).unwrap();
/// println!("r: {:?}", r);
/// ```
pub fn skew_rands(alpha: impl Into<f64>, n: usize) -> XResult<Vec<f64>> {
    let alpha: f64 = alpha.into();
    if alpha <= 0.0 || alpha >= 1.0 || alpha.is_nan() {
        return Err(StableError::InvalidSkewIndex.into());
    }
    let constants = StableConstants::new(alpha, 1.0);
    Ok((0..n)
        .into_par_iter()
        .map_init(
            || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
            |r, _| sample_standard_alpha_with_constants(&constants, alpha, r),
        )
        .collect())
}

/// Sample the symmetric standard Lévy stable distribution random number
///
/// # Arguments
///
/// * `alpha` - The index of stability, must be in the range (0, 2].
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::sym_standard_rand;
///
/// let alpha = 0.7;
/// let r = sym_standard_rand(alpha).unwrap();
/// println!("r: {}", r);
/// ```
pub fn sym_standard_rand(alpha: impl Into<f64>) -> XResult<f64> {
    let sym = SymmetricStandardStable::new(alpha)?;
    let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
    Ok(rng.sample(sym))
}

/// Sample the symmetric standard Lévy stable distribution random numbers
///
/// # Arguments
///
/// * `alpha` - The index of stability, must be in the range (0, 2].
/// * `n` - The number of samples to generate, must be greater than 0.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::sym_standard_rands;
///
/// let alpha = 0.7;
/// let n = 10;
/// let r = sym_standard_rands(alpha, n).unwrap();
/// println!("r: {:?}", r);
/// ```
pub fn sym_standard_rands(alpha: impl Into<f64>, n: usize) -> XResult<Vec<f64>> {
    let alpha: f64 = alpha.into();
    if alpha <= 0.0 || alpha > 2.0 || alpha.is_nan() {
        return Err(StableError::InvalidIndex.into());
    }
    if (alpha - 1.0).abs() < 1e-10 {
        Ok((0..n)
            .into_par_iter()
            .map_init(
                || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                |r, _| sample_standard_alpha_one(alpha, 0.0, r),
            )
            .collect())
    } else {
        let constants = StableConstants::new(alpha, 0.0);
        Ok((0..n)
            .into_par_iter()
            .map_init(
                || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                |r, _| sample_standard_alpha_with_constants(&constants, alpha, r),
            )
            .collect())
    }
}

/// Add two independent Lévy stable distributions
///
/// # Arguments
///
/// * `self` - The first Lévy stable distribution
/// * `other` - The second Lévy stable distribution
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::Stable;
///
/// let a = Stable::new(0.7, 1.0, 1.0, 0.0).unwrap();
/// let b = Stable::new(0.7, 1.0, 1.0, 0.0).unwrap();
/// let c = a + b;
/// ```
impl<T> Add<T> for Stable
where
    T: Into<f64>,
{
    type Output = Stable;

    fn add(self, other: T) -> Self::Output {
        Stable::new(self.alpha, self.beta, self.sigma, self.mu + other.into()).unwrap()
    }
}

/// Add a Lévy stable distribution and a f64
///
/// # Arguments
///
/// * `self` - The Lévy stable distribution.
/// * `other` - The constant.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::Stable;
///
/// let a = Stable::new(0.7, 1.0, 1.0, 0.0).unwrap();
/// let b = 1.0;
/// let c = a + b;
/// ```
impl Add<Stable> for f64 {
    type Output = Stable;

    fn add(self, other: Stable) -> Self::Output {
        Stable::new(other.alpha, other.beta, other.sigma, other.mu + self).unwrap()
    }
}

/// Add a i32 and a Lévy stable distribution
///
/// # Arguments
///
/// * `self` - The i32.
/// * `other` - The Lévy stable distribution.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::Stable;
///
/// let a = 1.0;
/// let b = Stable::new(0.7, 1.0, 1.0, 0.0).unwrap();
/// let c = a + b;
/// ```
impl Add<Stable> for i32 {
    type Output = Stable;

    fn add(self, other: Stable) -> Self::Output {
        let self_f64: f64 = self.into();
        Stable::new(other.alpha, other.beta, other.sigma, other.mu + self_f64).unwrap()
    }
}

/// Add a standard Lévy stable distribution and a number that can be converted to f64
///
/// # Arguments
///
/// * `self` - The standard Lévy stable distribution.
/// * `other` - The number.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::Stable;
///
/// let a = Stable::new(0.7, 1.0, 1.0, 0.0).unwrap();
/// let b = 1.0;
/// let c = a + b;
/// ```
impl<T> Add<T> for StandardStable
where
    T: Into<f64>,
{
    type Output = Stable;

    fn add(self, other: T) -> Self::Output {
        let other_f64: f64 = other.into();
        Stable::new(self.alpha, self.beta, 0.0, other_f64).unwrap()
    }
}

/// Add a standard Lévy stable distribution and a number that can be converted to f64
///
/// # Arguments
///
/// * `self` - The standard Lévy stable distribution.
/// * `other` - The number.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::Stable;
///
/// let a = Stable::new(0.7, 1.0, 1.0, 0.0).unwrap();
/// let b = 1.0;
/// let c = a + b;
/// ```
impl Add<StandardStable> for f64 {
    type Output = Stable;
    fn add(self, other: StandardStable) -> Self::Output {
        Stable::new(other.alpha, other.beta, 0.0, self).unwrap()
    }
}

/// Add a standard Lévy stable distribution and an i32
///
/// # Arguments
///
/// * `self` - The standard Lévy stable distribution.
/// * `other` - The number.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::Stable;
///
/// let a = Stable::new(0.7, 1.0, 1.0, 0.0).unwrap();
/// let b = 1;
/// let c = a + b;
/// ```
impl Add<StandardStable> for i32 {
    type Output = Stable;

    fn add(self, other: StandardStable) -> Self::Output {
        let self_f64: f64 = self.into();
        Stable::new(other.alpha, other.beta, 0.0, self_f64).unwrap()
    }
}

/// Multiply a Lévy stable distribution and a number that can be converted to f64
///
/// # Arguments
///
/// * `self` - The Lévy stable distribution.
/// * `other` - The number.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::Stable;
///
/// let a = Stable::new(0.7, 1.0, 1.0, 0.0).unwrap();
/// let b = 1.0;
/// let c = a * b;
/// ```
impl<T> Mul<T> for Stable
where
    T: Into<f64>,
{
    type Output = Stable;

    fn mul(self, other: T) -> Self::Output {
        let other_f64: f64 = other.into();
        let sigma: f64 = self.sigma * other_f64.abs();
        Stable::new(self.alpha, self.beta, sigma, self.mu).unwrap()
    }
}

/// Multiply a Lévy stable distribution and a number that can be converted to f64
///
/// # Arguments
///
/// * `self` - The Lévy stable distribution.
/// * `other` - The number.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::Stable;
///
/// let a = Stable::new(0.7, 1.0, 1.0, 0.0).unwrap();
/// let b = 1.0;
/// let c = a * b;
/// ```
impl Mul<Stable> for f64 {
    type Output = Stable;

    fn mul(self, other: Stable) -> Self::Output {
        let sigma: f64 = self * other.sigma.abs();
        Stable::new(other.alpha, other.beta, sigma, other.mu).unwrap()
    }
}

/// Multiply a Lévy stable distribution and an i32
///
/// # Arguments
///
/// * `self` - The Lévy stable distribution.
/// * `other` - The number.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::Stable;
///
/// let a = Stable::new(0.7, 1.0, 1.0, 0.0).unwrap();
/// let b = 1;
/// let c = a * b;
/// ```
impl Mul<Stable> for i32 {
    type Output = Stable;

    fn mul(self, other: Stable) -> Self::Output {
        let self_f64: f64 = self.into();
        let sigma: f64 = self_f64 * other.sigma.abs();
        Stable::new(other.alpha, other.beta, sigma, other.mu).unwrap()
    }
}

/// Multiply a standard Lévy stable distribution and a number that can be converted to f64
///
/// # Arguments
///
/// * `self` - The standard Lévy stable distribution.
/// * `other` - The number.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::Stable;
///
/// let a = Stable::new(0.7, 1.0, 1.0, 0.0).unwrap();
/// let b = 1.0;
/// let c = a * b;
/// ```
impl<T> Mul<T> for StandardStable
where
    T: Into<f64>,
{
    type Output = Stable;

    fn mul(self, other: T) -> Self::Output {
        let other_f64: f64 = other.into();
        Stable::new(self.alpha, self.beta, other_f64, 0.0).unwrap()
    }
}

/// Multiply a standard Lévy stable distribution and a number that can be converted to f64
///
/// # Arguments
///
/// * `self` - The standard Lévy stable distribution.
/// * `other` - The number.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::Stable;
///
/// let a = Stable::new(0.7, 1.0, 1.0, 0.0).unwrap();
/// let b = 1.0;
/// let c = a * b;
/// ```
impl Mul<StandardStable> for f64 {
    type Output = Stable;
    fn mul(self, other: StandardStable) -> Self::Output {
        Stable::new(other.alpha, other.beta, self, 0.0).unwrap()
    }
}

/// Multiply a standard Lévy stable distribution and an i32
///
/// # Arguments
///
/// * `self` - The standard Lévy stable distribution.
/// * `other` - The number.
///
/// # Example
///
/// ```rust
/// use diffusionx::random::stable::Stable;
///
/// let a = Stable::new(0.7, 1.0, 1.0, 0.0).unwrap();
/// let b = 1;
/// let c = a * b;
/// ```
impl Mul<StandardStable> for i32 {
    type Output = Stable;

    fn mul(self, other: StandardStable) -> Self::Output {
        let self_f64: f64 = self.into();
        Stable::new(other.alpha, other.beta, self_f64, 0.0).unwrap()
    }
}

/// Zolotariev's (M)-parameterization from mu to mu0
///
/// # Arguments
///
/// * `alpha` - The index of stability.
/// * `beta` - The skewness parameter.
/// * `sigma` - The scale parameter.
/// * `mu0` - The parameter of the Zolotariev's (M)-parameterization.
#[allow(dead_code)]
fn zolotariev(alpha: f64, beta: f64, sigma: f64, mu0: f64) -> f64 {
    if alpha != 1.0 {
        mu0 - beta * sigma * (PI * alpha / 2.0).tan()
    } else {
        mu0 - beta * sigma * 2.0 * sigma.ln() / PI
    }
}

/// Zolotariev's (M)-parameterization from mu to mu0
///
/// # Arguments
///
/// * `alpha` - The index of stability.
/// * `beta` - The skewness parameter.
/// * `sigma` - The scale parameter.
/// * `mu` - The location parameter.
#[allow(dead_code)]
fn zolotariev_inv(alpha: f64, beta: f64, sigma: f64, mu: f64) -> f64 {
    if alpha != 1.0 {
        mu + beta * sigma * (PI * alpha / 2.0).tan()
    } else {
        mu + beta * sigma * 2.0 * sigma.ln() / PI
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rng;

    #[test]
    fn test_sample_standard_alpha() {
        let alpha = 0.7;
        let beta = 1.0;
        let mut rng = rng();
        let standard = StandardStable::new(alpha, beta).unwrap();
        let r = rng.sample(standard);
        assert!(r.is_finite())
    }

    #[test]
    fn test_sample_symmetric_standard_alpha() {
        let alpha = 0.7;
        let mut rng = rng();
        let r = rng.sample(SymmetricStandardStable::new(alpha).unwrap());
        assert!(r.is_finite());
    }

    #[test]
    fn test_sample_symmetric_standard_alpha_rands() {
        let alpha = 0.7;
        let n = 10;
        let r = sym_standard_rands(alpha, n).unwrap();
        assert!(r.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_sample_skew_standard_alpha() {
        let alpha = 0.7;
        let mut rng = rng();
        let r = rng.sample(StandardSkewStable::new(alpha).unwrap());
        assert!(r > 0.0);
    }

    #[test]
    fn test_sample_skew_standard_alpha_rands() {
        let alpha = 0.7;
        let n = 10;
        let r = skew_rands(alpha, n).unwrap();
        assert!(r.iter().all(|&x| x > 0.0));
    }
}
