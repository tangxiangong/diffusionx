use crate::{XError, XResult, random::normal};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use realfft::{FftNum, RealFftPlanner, num_complex::Complex};
use std::{marker::PhantomData, ops::MulAssign};

/// Circulant embedding method for generating stationary Gaussian random fields with given correlation functions
pub struct CirculantEmbedding<F: Fn(T) -> T, T: Float = f64> {
    /// Number of grid points
    size: usize,
    /// Correlation function, takes distance as input and returns correlation
    correlation_fn: F,
    /// Phantom data
    _phantom: PhantomData<T>,
}

impl<F: Fn(T) -> T + Send + Sync, T: Float + Send + Sync> CirculantEmbedding<F, T> {
    /// Create a new `CirculantEmbedding`
    ///
    /// # Arguments
    ///
    /// * `size` - Number of grid points per dimension
    /// * `correlation_fn` - Correlation function, takes distance as input and returns correlation
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::utils::CirculantEmbedding;
    ///
    /// let embedding = CirculantEmbedding::new(100, |r| (-r / 10.0).exp());
    /// ```
    pub fn new(size: usize, correlation_fn: F) -> Self {
        CirculantEmbedding {
            size,
            correlation_fn,
            _phantom: PhantomData,
        }
    }

    /// Generate a one-dimensional stationary Gaussian random field
    pub fn generate(&self) -> XResult<Vec<T>>
    where
        T: FftNum + MulAssign<T>,
        StandardNormal: Distribution<T>,
    {
        let n = self.size;
        let m = 2 * n;

        let spectrum_len = m / 2 + 1;

        let mut first_row: Vec<_> = (0..m)
            .into_par_iter()
            .map(|i| {
                let dist = if i <= m / 2 {
                    T::from(i).unwrap()
                } else {
                    T::from(m - i).unwrap()
                };
                (self.correlation_fn)(dist)
            })
            .collect();

        let mut planner = RealFftPlanner::new();

        let fft = planner.plan_fft_forward(m);
        let mut eigenvalues = fft.make_output_vec();
        fft.process(&mut first_row, &mut eigenvalues)?;

        let sqrt_eigenvalues = eigenvalues
            .iter()
            .map(|val| {
                if val.re < T::zero() {
                    return Err(XError::NotPositiveDefinite(val.re.to_f64().unwrap()));
                }
                Ok(Complex::new(val.re.sqrt(), T::zero()))
            })
            .collect::<XResult<Vec<Complex<_>>>>()?;

        let ifft = planner.plan_fft_inverse(m);

        let z_real = normal::standard_rands::<T>(spectrum_len);
        let z_imag = normal::standard_rands::<T>(spectrum_len);

        let mut spectrum: Vec<Complex<_>> = (0..spectrum_len)
            .map(|i| {
                let sqrt_lambda = sqrt_eigenvalues[i].re;
                if i == 0 || (i == m / 2 && m.is_multiple_of(2)) {
                    // DC and Nyquist components must be real
                    Complex::new(sqrt_lambda * z_real[i], T::zero())
                } else {
                    Complex::new(sqrt_lambda * z_real[i], sqrt_lambda * z_imag[i])
                }
            })
            .collect();

        let mut result = ifft.make_output_vec();
        ifft.process(&mut spectrum, &mut result)?;
        let scale = T::one() / T::from(m).unwrap();
        result.iter_mut().for_each(|x| *x *= scale);
        result.truncate(n);

        Ok(result)
    }
}
