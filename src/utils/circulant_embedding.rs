use crate::{XError, XResult, random::normal};
use rayon::prelude::*;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex, num_complex::Complex};
use std::sync::{Arc, LazyLock, Mutex};

// Use a global cached FFT planner to avoid repeated creation
static REAL_FFT_PLANNER: LazyLock<Mutex<RealFftPlanner<f64>>> =
    LazyLock::new(|| Mutex::new(RealFftPlanner::new()));

/// Circulant embedding method for generating stationary Gaussian random fields with given correlation functions
pub struct CirculantEmbedding {
    /// Number of grid points per dimension
    size: usize,
    /// Correlation function, takes distance as input and returns correlation
    correlation_fn: Box<dyn Fn(f64) -> f64 + Send + Sync + 'static>,
    /// Cache of the first row of the circulant embedding matrix
    first_row_cache: Option<Vec<f64>>,
    /// Cache of the square roots of eigenvalues (for faster generation)
    sqrt_eigenvalues_cache: Option<Vec<Complex<f64>>>,
    /// Cache of the forward FFT plan
    fft_forward_plan: Option<Arc<dyn RealToComplex<f64>>>,
    /// Cache of the inverse FFT plan of the circulant embedding matrix
    fft_inverse_plan: Option<Arc<dyn ComplexToReal<f64>>>,
}

impl CirculantEmbedding {
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
    pub fn new<F>(size: usize, correlation_fn: F) -> Self
    where
        F: Fn(f64) -> f64 + Send + Sync + 'static,
    {
        CirculantEmbedding {
            size,
            correlation_fn: Box::new(correlation_fn),
            first_row_cache: None,
            sqrt_eigenvalues_cache: None,
            fft_forward_plan: None,
            fft_inverse_plan: None,
        }
    }

    /// Precompute and cache the first row of the circulant embedding matrix,
    /// its eigenvalues (via FFT), and the inverse FFT plan.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::utils::CirculantEmbedding;
    ///
    /// let embedding = CirculantEmbedding::new(100, |r| (-r / 10.0).exp());
    /// let embedding = CirculantEmbedding::new(100, |r| (-r / 10.0).exp());
    /// embedding.precompute_correlation().unwrap();
    /// ```
    pub fn precompute_correlation(&mut self) -> XResult<()> {
        let n = self.size;
        let m = 2 * n;

        // Build the first row of the circulant embedding matrix
        let mut first_row: Vec<f64> = (0..m)
            .into_par_iter()
            .map(|i| {
                let dist = if i <= m / 2 { i as f64 } else { (m - i) as f64 };
                (self.correlation_fn)(dist)
            })
            .collect();

        // Plan and execute forward FFT to get eigenvalues
        let mut eigenvalues: Vec<Complex<f64>>;
        let fft_forward: Arc<dyn RealToComplex<f64>>;
        {
            let mut planner = REAL_FFT_PLANNER
                .lock()
                .map_err(|_| XError::FFTPlannerLock)?;
            fft_forward = planner.plan_fft_forward(m);
            eigenvalues = fft_forward.make_output_vec();
            fft_forward
                .process(&mut first_row, &mut eigenvalues)
                .map_err(|e| XError::Other(format!("FFT processing error: {:?}", e)))?;
        }

        // Check if all eigenvalues are positive and compute square roots
        let mut sqrt_eigenvalues = Vec::with_capacity(eigenvalues.len());
        for val in &eigenvalues {
            if val.re < -1e-10 {
                // Clear potentially invalid caches and return error
                self.first_row_cache = None;
                self.sqrt_eigenvalues_cache = None;
                self.fft_forward_plan = None;
                self.fft_inverse_plan = None;
                return Err(XError::NotPositiveDefinite(val.re));
            }
            // Precompute square root of eigenvalue for faster generation
            let sqrt_lambda = val.re.max(0.0).sqrt();
            sqrt_eigenvalues.push(Complex::new(sqrt_lambda, 0.0));
        }

        // Plan and cache inverse FFT plan
        let fft_inverse: Arc<dyn ComplexToReal<f64>>;
        {
            let mut planner = REAL_FFT_PLANNER
                .lock()
                .map_err(|_| XError::FFTPlannerLock)?;
            fft_inverse = planner.plan_fft_inverse(m);
        }

        // Cache successful results
        self.first_row_cache = None; // Don't need to keep the row
        self.sqrt_eigenvalues_cache = Some(sqrt_eigenvalues);
        self.fft_forward_plan = Some(fft_forward);
        self.fft_inverse_plan = Some(fft_inverse);

        Ok(())
    }

    /// Generate a one-dimensional stationary Gaussian random field
    pub fn generate(&self) -> XResult<Vec<f64>> {
        let n = self.size;
        let m = 2 * n;
        let spectrum_len = m / 2 + 1;

        // Get sqrt eigenvalues and inverse FFT plan, either from cache or by computing on the fly
        let (mut sqrt_eigenvalues, ifft): (Vec<Complex<f64>>, Arc<dyn ComplexToReal<f64>>);

        if let (Some(cached_sqrt_eigenvalues), Some(cached_ifft_plan)) =
            (&self.sqrt_eigenvalues_cache, &self.fft_inverse_plan)
        {
            // Use cached sqrt eigenvalues and inverse plan
            sqrt_eigenvalues = cached_sqrt_eigenvalues.clone();
            ifft = cached_ifft_plan.clone();
        } else {
            // Compute on the fly if not precomputed
            let mut first_row: Vec<f64> = (0..m)
                .into_par_iter()
                .map(|i| {
                    let dist = if i <= m / 2 { i as f64 } else { (m - i) as f64 };
                    (self.correlation_fn)(dist)
                })
                .collect();

            // Calculate the eigenvalues (using real FFT)
            let mut eigenvalues: Vec<Complex<f64>>;
            let fft_forward: Arc<dyn RealToComplex<f64>>;
            {
                let mut planner = REAL_FFT_PLANNER
                    .lock()
                    .map_err(|_| XError::FFTPlannerLock)?;
                fft_forward = planner.plan_fft_forward(m);
                eigenvalues = fft_forward.make_output_vec();
                fft_forward
                    .process(&mut first_row, &mut eigenvalues)
                    .map_err(|e| XError::Other(format!("FFT processing error: {:?}", e)))?;
            }

            // Check if all eigenvalues are positive and compute square roots
            sqrt_eigenvalues = Vec::with_capacity(eigenvalues.len());
            for val in &eigenvalues {
                if val.re < -1e-10 {
                    return Err(XError::NotPositiveDefinite(val.re));
                }
                let sqrt_lambda = val.re.max(0.0).sqrt();
                sqrt_eigenvalues.push(Complex::new(sqrt_lambda, 0.0));
            }

            // Get inverse FFT plan
            ifft = {
                let mut planner = REAL_FFT_PLANNER
                    .lock()
                    .map_err(|_| XError::FFTPlannerLock)?;
                planner.plan_fft_inverse(m)
            };
        }

        // Generate random Gaussian noise for the spectrum
        // For real FFT output of length m/2+1, we need special handling
        let z_real = normal::standard_rands::<f64>(spectrum_len);
        let z_imag = normal::standard_rands::<f64>(spectrum_len);

        // Build the complex spectrum Y = sqrt(Lambda) * Z
        let mut spectrum: Vec<Complex<f64>> = (0..spectrum_len)
            .map(|i| {
                let sqrt_lambda = sqrt_eigenvalues[i].re;
                if i == 0 || (i == m / 2 && m.is_multiple_of(2)) {
                    // DC and Nyquist components must be real
                    Complex::new(sqrt_lambda * z_real[i], 0.0)
                } else {
                    Complex::new(sqrt_lambda * z_real[i], sqrt_lambda * z_imag[i])
                }
            })
            .collect();

        // Execute inverse FFT to get real-valued result
        let mut result = ifft.make_output_vec();
        ifft.process(&mut spectrum, &mut result)
            .map_err(|e| XError::Other(format!("Inverse FFT processing error: {:?}", e)))?;

        // Normalize (IFFT result needs scaling by 1/m) and take first n elements
        let scale = 1.0 / m as f64;
        result.iter_mut().for_each(|x| *x *= scale);
        result.truncate(n);

        Ok(result)
    }

    /// Generate and normalize a one-dimensional stationary Gaussian random field
    pub fn generate_normalized(&self) -> XResult<Vec<f64>> {
        let mut result = self.generate()?;
        let n = result.len();

        // Normalize the result, ensuring the mean is 0 and the variance is 1
        let mut sum = 0.0;
        for &x in &result {
            sum += x;
        }
        let mean = sum / n as f64;

        let mut sum_squared_diff = 0.0;
        for &x in &result {
            sum_squared_diff += (x - mean).powi(2);
        }
        let variance = sum_squared_diff / n as f64;

        if variance > 1e-10 {
            let scale_factor = 1.0 / variance.sqrt();
            result
                .par_iter_mut()
                .for_each(|x| *x = (*x - mean) * scale_factor);
        }

        Ok(result)
    }

    /// Generate multiple independent random fields
    pub fn generate_multiple(&self, count: usize) -> XResult<Vec<Vec<f64>>> {
        (0..count)
            .into_par_iter()
            .map(|_| self.generate())
            .collect()
    }

    /// Generate multiple independent normalized random fields
    pub fn generate_multiple_normalized(&self, count: usize) -> XResult<Vec<Vec<f64>>> {
        (0..count)
            .into_par_iter()
            .map(|_| self.generate_normalized())
            .collect()
    }
}

/// Fractional Brownian motion correlation function
pub fn fbm_correlation(hurst: f64, time_step: f64) -> impl Fn(f64) -> f64 {
    move |r: f64| {
        let r_abs = r.abs();
        if r_abs < 1e-10 {
            return 1.0;
        }

        let h2 = 2.0 * hurst;
        0.5 * time_step.powf(h2)
            * ((r_abs + 1.0).powf(h2) - 2.0 * r_abs.powf(h2) + (r_abs - 1.0).abs().powf(h2))
    }
}

/// Exponential correlation function: exp(-r/l)
pub fn exponential_correlation(length_scale: f64) -> impl Fn(f64) -> f64 {
    move |r: f64| (-r / length_scale).exp()
}

/// Gaussian correlation function: exp(-(r/l)^2)
pub fn gaussian_correlation(length_scale: f64) -> impl Fn(f64) -> f64 {
    let inv_length_scale_sq = 1.0 / (length_scale * length_scale);
    move |r: f64| (-r * r * inv_length_scale_sq).exp()
}

/// Matérn correlation function (nu=1/2): exp(-r/l)
pub fn matern_half_correlation(length_scale: f64) -> impl Fn(f64) -> f64 {
    exponential_correlation(length_scale)
}

/// Matérn correlation function (nu=3/2): (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)
pub fn matern_three_half_correlation(length_scale: f64) -> impl Fn(f64) -> f64 {
    let sqrt3 = 3.0_f64.sqrt();
    let factor = sqrt3 / length_scale;
    move |r: f64| {
        let scaled_r = r * factor;
        (1.0 + scaled_r) * (-scaled_r).exp()
    }
}

/// Matérn correlation function (nu=5/2): (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
pub fn matern_five_half_correlation(length_scale: f64) -> impl Fn(f64) -> f64 {
    let sqrt5 = 5.0_f64.sqrt();
    let factor = sqrt5 / length_scale;
    let factor_sq = 5.0 / (3.0 * length_scale * length_scale);
    move |r: f64| {
        let scaled_r = r * factor;
        (1.0 + scaled_r + r * r * factor_sq) * (-scaled_r).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::E;

    #[test]
    #[ignore]
    fn test_gaussian_field() {
        let size = 64;
        let length_scale = 10.0;
        let corr_fn = gaussian_correlation(length_scale);
        let embedding = CirculantEmbedding::new(size, corr_fn);

        let field = embedding
            .generate_normalized()
            .expect("Field generation failed");

        // Check the size of the generated field
        assert_eq!(field.len(), size);

        // Calculate the sample mean (should be close to 0)
        let mean = field.iter().sum::<f64>() / size as f64;
        assert!(mean.abs() < 0.5, "Mean {mean} is not in the expected range");

        // Calculate the sample variance (should be close to 1)
        let variance = field.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / size as f64;
        assert!(
            (variance - 1.0).abs() < 1.0,
            "Variance {variance} is not in the expected range"
        );
    }

    #[test]
    fn test_precompute_correlation() {
        let size = 64;
        let length_scale = 10.0;
        let corr_fn = gaussian_correlation(length_scale);
        let mut embedding = CirculantEmbedding::new(size, corr_fn);

        embedding
            .precompute_correlation()
            .expect("Precompute failed");
        let field1 = embedding.generate().expect("Field generation failed");
        let field2 = embedding.generate().expect("Field generation failed");

        assert_eq!(field1.len(), size);
        assert_eq!(field2.len(), size);
    }

    #[test]
    fn test_correlation_functions() {
        let length_scale = 2.0;

        // Test the exponential correlation function
        let exp_corr = exponential_correlation(length_scale);
        assert_eq!(exp_corr(0.0), 1.0);
        assert!((exp_corr(length_scale) - 1.0 / E).abs() < 1e-10);

        // Test the Gaussian correlation function
        let gauss_corr = gaussian_correlation(length_scale);
        assert_eq!(gauss_corr(0.0), 1.0);
        assert!((gauss_corr(length_scale) - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    #[ignore]
    fn test_generate_multiple() {
        let size = 32;
        let length_scale = 5.0;
        let corr_fn = exponential_correlation(length_scale);
        let embedding = CirculantEmbedding::new(size, corr_fn);

        let count = 5;
        let fields = embedding
            .generate_multiple(count)
            .expect("Generate multiple fields failed");

        assert_eq!(fields.len(), count);
        for field in &fields {
            assert_eq!(field.len(), size);
        }
    }
}
