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

use crate::{FloatExt, StableError, XResult, random::STABLE_PAR_THRESHOLD};
use rand::{Rng, prelude::*};
use rand_distr::{Exp1, uniform::SampleUniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// Precomputed constants for stable distribution sampling
#[derive(Debug, Clone, Copy)]
pub(crate) struct StableConstants<T: FloatExt = f64> {
    /// 1.0 / alpha
    inv_alpha: T,
    /// (1.0 - alpha) / alpha
    one_minus_alpha_div_alpha: T,
    /// atan(beta * tan(alpha * PI/2)) / alpha
    b: T,
    /// (1.0 + (beta * tan(alpha * PI/2))^2)^(1/(2*alpha))
    s: T,
}

impl<T: FloatExt> StableConstants<T> {
    #[inline]
    pub fn new(alpha: T, beta: T) -> Self {
        let inv_alpha = T::one() / alpha;
        let one_minus_alpha_div_alpha = (T::one() - alpha) * inv_alpha;
        let tmp = beta * (alpha * T::FRAC_PI_2()).tan();
        let b = tmp.atan() * inv_alpha;
        let s = (T::one() + tmp * tmp).powf(T::from(0.5).unwrap() * inv_alpha);
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
pub struct StandardStable<T: FloatExt = f64> {
    /// Index of stability
    alpha: T,
    /// Skewness parameter
    beta: T,
}

impl<T: FloatExt> StandardStable<T> {
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
    pub fn new(alpha: T, beta: T) -> XResult<Self> {
        if alpha <= T::zero() || alpha > T::from(2).unwrap() || alpha.is_nan() {
            return Err(StableError::InvalidIndex.into());
        }
        if !(-T::one()..=T::one()).contains(&beta) {
            return Err(StableError::InvalidSkewness.into());
        }
        Ok(Self { alpha, beta })
    }

    /// Get the index of stability
    pub fn get_index(&self) -> T {
        self.alpha
    }

    /// Get the skewness parameter
    pub fn get_skewness(&self) -> T {
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
    pub fn samples(&self, n: usize) -> XResult<Vec<T>>
    where
        T: SampleUniform,
        Exp1: Distribution<T>,
    {
        standard_rands(self.alpha, self.beta, n)
    }
}

/// Sample standard stable random number when alpha is not 1
pub(crate) fn sample_standard_alpha<T: FloatExt + SampleUniform, R: Rng + ?Sized>(
    alpha: T,
    beta: T,
    rng: &mut R,
) -> T
where
    Exp1: Distribution<T>,
{
    let constants = StableConstants::new(alpha, beta);
    sample_standard_alpha_with_constants(&constants, alpha, rng)
}

/// Sample standard stable random number with precomputed constants
#[inline]
pub(crate) fn sample_standard_alpha_with_constants<T, R: Rng + ?Sized>(
    c: &StableConstants<T>,
    alpha: T,
    rng: &mut R,
) -> T
where
    T: FloatExt + SampleUniform,
    Exp1: Distribution<T>,
{
    let v = rng.random_range(-T::FRAC_PI_2()..T::FRAC_PI_2());
    let w: T = rng.sample(Exp1);
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
pub(crate) fn sample_standard_alpha_one<T, R: Rng + ?Sized>(_alpha: T, beta: T, rng: &mut R) -> T
where
    T: FloatExt + SampleUniform,
    Exp1: Distribution<T>,
{
    let v = rng.random_range(-T::FRAC_PI_2()..T::FRAC_PI_2());
    let w: T = rng.sample(Exp1);
    let half_pi_plus_beta_v = T::FRAC_PI_2() + beta * v;
    let c1 = half_pi_plus_beta_v * v.tan();
    let c2 = ((T::FRAC_PI_2() * w * v.cos()) / half_pi_plus_beta_v).ln() * beta;
    (c1 - c2) * T::FRAC_2_PI()
}

/// Sample from the standard Lévy stable distribution
impl<T: FloatExt + SampleUniform> Distribution<T> for StandardStable<T>
where
    Exp1: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        if (self.alpha - T::one()).abs() > T::epsilon() {
            sample_standard_alpha(self.alpha, self.beta, rng)
        } else {
            sample_standard_alpha_one(self.alpha, self.beta, rng)
        }
    }
}

/// Lévy stable distribution
#[derive(Debug, Clone, Copy)]
pub struct Stable<T: FloatExt = f64> {
    /// Index of stability
    alpha: T,
    /// Skewness parameter
    beta: T,
    /// Scale parameter
    sigma: T,
    /// Location parameter
    mu: T,
}

/// Convert a standard Lévy stable distribution to a Lévy stable distribution
impl<T: FloatExt> From<&Stable<T>> for StandardStable<T> {
    fn from(stable: &Stable<T>) -> Self {
        StandardStable::new(stable.alpha, stable.beta).unwrap()
    }
}

impl<T: FloatExt> Stable<T> {
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
    pub fn new(alpha: T, beta: T, sigma: T, mu: T) -> XResult<Self> {
        if alpha <= T::zero() || alpha > T::from(2).unwrap() || alpha.is_nan() {
            return Err(StableError::InvalidIndex.into());
        }
        if !(-T::one()..=T::one()).contains(&beta) {
            return Err(StableError::InvalidSkewness.into());
        }
        if sigma <= T::zero() || sigma.is_nan() {
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
    pub fn get_index(&self) -> T {
        self.alpha
    }

    /// Get the skewness parameter
    pub fn get_skewness(&self) -> T {
        self.beta
    }

    /// Get the scale parameter
    pub fn get_scale(&self) -> T {
        self.sigma
    }

    /// Get the location parameter
    pub fn get_location(&self) -> T {
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
    pub fn samples(&self, n: usize) -> XResult<Vec<T>>
    where
        T: SampleUniform,
        Exp1: Distribution<T>,
    {
        rands(self.alpha, self.beta, self.sigma, self.mu, n)
    }
}

/// Sample from the Lévy stable distribution
impl<T: FloatExt + SampleUniform> Distribution<T> for Stable<T>
where
    Exp1: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        let standard = StandardStable::from(self);
        let r = rng.sample(standard);
        if self.alpha != T::one() {
            self.sigma * r + self.mu
        } else {
            self.sigma * r
                + self.mu
                + T::from(2).unwrap() * self.beta * self.sigma * self.sigma.ln() / T::PI()
        }
    }
}

/// Standard skew Lévy stable distribution
#[derive(Debug, Clone, Copy)]
pub struct StandardSkewStable<T: FloatExt = f64>(pub T);

impl<T: FloatExt> StandardSkewStable<T> {
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
    pub fn new(alpha: T) -> XResult<Self> {
        if alpha <= T::zero() || alpha >= T::one() || alpha.is_nan() {
            return Err(StableError::InvalidSkewIndex.into());
        }
        Ok(Self(alpha))
    }

    /// Get the index of stability
    pub fn get_index(&self) -> T {
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
    pub fn samples(&self, n: usize) -> XResult<Vec<T>>
    where
        T: SampleUniform,
        Exp1: Distribution<T>,
    {
        skew_rands(self.0, n)
    }
}

/// Sample standard skew stable random number
///
/// # Panic
///
/// if the skew index is invalid
impl<T: FloatExt + SampleUniform> Distribution<T> for StandardSkewStable<T>
where
    Exp1: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        let alpha = self.0;
        if alpha <= T::zero() || alpha >= T::one() || alpha.is_nan() {
            panic!("Invalid skew index");
        }
        sample_standard_alpha(self.0, T::one(), rng)
    }
}

/// Symmetric Lévy stable distribution
#[derive(Debug, Clone, Copy)]
pub struct SymmetricStandardStable<T: FloatExt = f64>(pub T);

impl<T: FloatExt> SymmetricStandardStable<T> {
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
    pub fn new(alpha: T) -> XResult<Self> {
        if alpha <= T::zero() || alpha >= T::from(2).unwrap() || alpha.is_nan() {
            return Err(StableError::InvalidSkewIndex.into());
        }
        Ok(Self(alpha))
    }

    /// Get the index of stability
    pub fn get_index(&self) -> T {
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
    pub fn samples(&self, n: usize) -> XResult<Vec<T>>
    where
        T: SampleUniform,
        Exp1: Distribution<T>,
    {
        sym_standard_rands(self.0, n)
    }
}

/// Sample symmetric standard stable random number
///
/// # Panic
///
/// if the stability index is invalid
impl<T: FloatExt + SampleUniform> Distribution<T> for SymmetricStandardStable<T>
where
    Exp1: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        let alpha = self.0;
        if alpha <= T::zero() || alpha > T::from(2).unwrap() || alpha.is_nan() {
            panic!("Invalid stability index");
        }
        if (alpha - T::one()).abs() > T::epsilon() {
            let inv_alpha = T::one() / alpha;
            let one_minus_alpha_div_alpha = (T::one() - alpha) * inv_alpha;
            sample_sym_standard_alpha_with_constants(
                inv_alpha,
                one_minus_alpha_div_alpha,
                alpha,
                rng,
            )
        } else {
            sample_sym_standard_alpha_one(rng)
        }
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
pub fn standard_rand<T: FloatExt + SampleUniform>(alpha: T, beta: T) -> XResult<T>
where
    Exp1: Distribution<T>,
{
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
pub fn standard_rands<T>(alpha: T, beta: T, n: usize) -> XResult<Vec<T>>
where
    T: FloatExt + SampleUniform,
    Exp1: Distribution<T>,
{
    if alpha <= T::zero() || alpha > T::from(2).unwrap() || alpha.is_nan() {
        return Err(StableError::InvalidIndex.into());
    }
    if !(-T::one()..=T::one()).contains(&beta) {
        return Err(StableError::InvalidSkewness.into());
    }
    if (alpha - T::one()).abs() < T::epsilon() {
        if n <= STABLE_PAR_THRESHOLD {
            let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
            Ok((0..n)
                .map(|_| sample_standard_alpha_one(alpha, beta, &mut rng))
                .collect())
        } else {
            Ok((0..n)
                .into_par_iter()
                .map_init(
                    || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                    |r, _| sample_standard_alpha_one(alpha, beta, r),
                )
                .collect())
        }
    } else {
        let constants = StableConstants::new(alpha, beta);
        if n <= STABLE_PAR_THRESHOLD {
            let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
            Ok((0..n)
                .map(|_| sample_standard_alpha_with_constants(&constants, alpha, &mut rng))
                .collect())
        } else {
            Ok((0..n)
                .into_par_iter()
                .map_init(
                    || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                    |r, _| sample_standard_alpha_with_constants(&constants, alpha, r),
                )
                .collect())
        }
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
pub fn rand<T: FloatExt + SampleUniform>(alpha: T, beta: T, sigma: T, mu: T) -> XResult<T>
where
    Exp1: Distribution<T>,
{
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
pub fn rands<T>(alpha: T, beta: T, sigma: T, mu: T, n: usize) -> XResult<Vec<T>>
where
    T: FloatExt + SampleUniform,
    Exp1: Distribution<T>,
{
    let two = T::from(2).unwrap();
    if alpha <= T::zero() || alpha > two || alpha.is_nan() {
        return Err(StableError::InvalidIndex.into());
    }
    if !(-T::one()..=T::one()).contains(&beta) {
        return Err(StableError::InvalidSkewness.into());
    }
    if sigma <= T::zero() || sigma.is_nan() {
        return Err(StableError::InvalidScale.into());
    }
    if mu.is_nan() {
        return Err(StableError::InvalidLocation.into());
    }
    if (alpha - T::one()).abs() < T::epsilon() {
        let correction = two * beta * sigma * sigma.ln() / T::PI();
        if n <= STABLE_PAR_THRESHOLD {
            let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
            Ok((0..n)
                .map(|_| {
                    let std_sample = sample_standard_alpha_one(alpha, beta, &mut rng);
                    sigma * std_sample + mu + correction
                })
                .collect())
        } else {
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
        }
    } else {
        let constants = StableConstants::new(alpha, beta);
        if n <= STABLE_PAR_THRESHOLD {
            let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
            Ok((0..n)
                .map(|_| {
                    let std_sample =
                        sample_standard_alpha_with_constants(&constants, alpha, &mut rng);
                    sigma * std_sample + mu
                })
                .collect())
        } else {
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
pub fn skew_rand<T: FloatExt + SampleUniform>(alpha: T) -> XResult<T>
where
    Exp1: Distribution<T>,
{
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
pub fn skew_rands<T>(alpha: T, n: usize) -> XResult<Vec<T>>
where
    T: FloatExt + SampleUniform,
    Exp1: Distribution<T>,
{
    if alpha <= T::zero() || alpha >= T::one() || alpha.is_nan() {
        return Err(StableError::InvalidSkewIndex.into());
    }
    let constants = StableConstants::new(alpha, T::one());
    if n <= STABLE_PAR_THRESHOLD {
        let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
        Ok((0..n)
            .map(|_| sample_standard_alpha_with_constants(&constants, alpha, &mut rng))
            .collect())
    } else {
        Ok((0..n)
            .into_par_iter()
            .map_init(
                || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                |r, _| sample_standard_alpha_with_constants(&constants, alpha, r),
            )
            .collect())
    }
}

/// Sample symmetric standard stable random number with precomputed constants
#[inline]
pub(crate) fn sample_sym_standard_alpha_with_constants<T, R: Rng + ?Sized>(
    inv_alpha: T,
    one_minus_alpha_div_alpha: T,
    alpha: T,
    rng: &mut R,
) -> T
where
    T: FloatExt + SampleUniform,
    Exp1: Distribution<T>,
{
    let v = rng.random_range(-T::FRAC_PI_2()..T::FRAC_PI_2());
    let w: T = rng.sample(Exp1);
    let cos_v = v.cos();
    // alpha * sin(v + b) / cos(v)^(1/alpha)
    let c1 = alpha * v.sin() / cos_v.powf(inv_alpha);
    // ((cos(v - alpha*(v+b)) / w))^((1-alpha)/alpha)
    let c2 = ((v - alpha * v).cos() / w).powf(one_minus_alpha_div_alpha);
    c1 * c2
}

/// Sample standard stable random number when alpha is 1
#[inline]
pub(crate) fn sample_sym_standard_alpha_one<T, R: Rng + ?Sized>(rng: &mut R) -> T
where
    T: FloatExt + SampleUniform,
    Exp1: Distribution<T>,
{
    let v = rng.random_range(-T::FRAC_PI_2()..T::FRAC_PI_2());
    v.tan()
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
pub fn sym_standard_rand<T: FloatExt + SampleUniform>(alpha: T) -> XResult<T>
where
    Exp1: Distribution<T>,
{
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
pub fn sym_standard_rands<T>(alpha: T, n: usize) -> XResult<Vec<T>>
where
    T: FloatExt + SampleUniform,
    Exp1: Distribution<T>,
{
    if alpha <= T::zero() || alpha > T::from(2).unwrap() || alpha.is_nan() {
        return Err(StableError::InvalidIndex.into());
    }
    if (alpha - T::one()).abs() < T::epsilon() {
        if n <= STABLE_PAR_THRESHOLD {
            let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
            Ok((0..n)
                .map(|_| sample_sym_standard_alpha_one(&mut rng))
                .collect())
        } else {
            Ok((0..n)
                .into_par_iter()
                .map_init(
                    || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                    |r, _| sample_sym_standard_alpha_one(r),
                )
                .collect())
        }
    } else {
        let inv_alpha = T::one() / alpha;
        let one_minus_alpha_div_alpha = (T::one() - alpha) * inv_alpha;
        if n <= STABLE_PAR_THRESHOLD {
            let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
            Ok((0..n)
                .map(|_| {
                    sample_sym_standard_alpha_with_constants(
                        inv_alpha,
                        one_minus_alpha_div_alpha,
                        alpha,
                        &mut rng,
                    )
                })
                .collect())
        } else {
            Ok((0..n)
                .into_par_iter()
                .map_init(
                    || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                    |r, _| {
                        sample_sym_standard_alpha_with_constants(
                            inv_alpha,
                            one_minus_alpha_div_alpha,
                            alpha,
                            r,
                        )
                    },
                )
                .collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Float;
    use rand::rng;

    #[test]
    fn test_sample_standard_alpha() {
        let alpha = 0.7;
        let beta = 1.0;
        let mut rng = rng();
        let standard = StandardStable::new(alpha, beta).unwrap();
        let r = rng.sample(standard);
        assert!(r.is_finite());
        let standard = StandardStable::new(alpha as f32, beta as f32).unwrap();
        let r = rng.sample(standard);
        assert!(r.is_finite());
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
