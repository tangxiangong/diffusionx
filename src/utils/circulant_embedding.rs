use crate::{XError, XResult, random::normal};
use once_cell::sync::Lazy;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::{Arc, Mutex};

// Use a global cached FFT planner to avoid repeated creation
static FFT_PLANNER: Lazy<Mutex<FftPlanner<f64>>> = Lazy::new(|| Mutex::new(FftPlanner::new()));

/// Circulant embedding method for generating stationary Gaussian random fields with given correlation functions
pub struct CirculantEmbedding {
    /// Number of grid points per dimension
    size: usize,
    /// Correlation function, takes distance as input and returns correlation
    correlation_fn: Box<dyn Fn(f64) -> f64 + Send + Sync + 'static>,
    /// Cache of the first row of the circulant embedding matrix
    first_row_cache: Option<Vec<f64>>,
    /// Cache of the eigenvalues of the circulant embedding matrix
    eigenvalues_cache: Option<Vec<Complex<f64>>>,
    /// Cache of the inverse FFT plan of the circulant embedding matrix
    fft_inverse_plan: Option<Arc<dyn Fft<f64>>>,
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
            eigenvalues_cache: None,
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
        let first_row: Vec<f64> = (0..m)
            .into_par_iter()
            .map(|i| {
                let dist = if i <= m / 2 { i as f64 } else { (m - i) as f64 };
                (self.correlation_fn)(dist)
            })
            .collect();

        let mut eigenvalues: Vec<Complex<f64>> =
            first_row.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Plan and execute forward FFT to get eigenvalues
        {
            let mut planner = FFT_PLANNER.lock().map_err(|_| XError::FFTPlannerLock)?;
            let fft = planner.plan_fft_forward(m);
            fft.process(&mut eigenvalues);
        }

        // Check if all eigenvalues are positive
        if let Some(negative_eigenvalue) = eigenvalues.iter().find(|val| val.re < -1e-10) {
            // Clear potentially invalid caches and return error
            self.first_row_cache = Some(first_row); // Cache the row anyway
            self.eigenvalues_cache = None;
            self.fft_inverse_plan = None;
            return Err(XError::NotPositiveDefinite(negative_eigenvalue.re));
        }

        // Plan and cache inverse FFT plan
        {
            let mut planner = FFT_PLANNER.lock().map_err(|_| XError::FFTPlannerLock)?; // Lock again
            self.fft_inverse_plan = Some(planner.plan_fft_inverse(m));
        }

        // Cache successful results
        self.first_row_cache = Some(first_row); // Keep cached row
        self.eigenvalues_cache = Some(eigenvalues); // Cache eigenvalues

        Ok(())
    }

    /// Generate a one-dimensional stationary Gaussian random field
    pub fn generate(&self) -> XResult<Vec<f64>> {
        let n = self.size;
        let m = 2 * n;

        // Get eigenvalues and inverse FFT plan, either from cache or by computing on the fly
        let (mut eigenvalues, ifft) = if let (Some(cached_eigenvalues), Some(cached_ifft_plan)) =
            (&self.eigenvalues_cache, &self.fft_inverse_plan)
        {
            // Use cached eigenvalues and inverse plan
            // Clone eigenvalues as they will be modified by subsequent steps
            // Clone Arc for the plan (cheap)
            (cached_eigenvalues.clone(), cached_ifft_plan.clone())
        } else {
            // Compute on the fly if not precomputed or precomputation failed
            let first_row = if let Some(ref cache) = self.first_row_cache {
                // Use cached first row if available (e.g., from failed precomputation)
                cache.clone()
            } else {
                // Compute first row if no cache exists
                (0..m)
                    .into_par_iter()
                    .map(|i| {
                        let dist = if i <= m / 2 { i as f64 } else { (m - i) as f64 };
                        (self.correlation_fn)(dist)
                    })
                    .collect()
            };

            // Convert to complex for FFT
            let mut complex_data: Vec<Complex<f64>> = first_row
                .into_iter()
                .map(|x| Complex::new(x, 0.0))
                .collect();

            // Calculate the eigenvalues (using FFT)
            {
                let mut planner = FFT_PLANNER.lock().map_err(|_| XError::FFTPlannerLock)?;
                let fft = planner.plan_fft_forward(m);
                fft.process(&mut complex_data);
            }

            // Check if all eigenvalues are positive
            if let Some(negative_eigenvalue) = complex_data.iter().find(|val| val.re < -1e-10) {
                return Err(XError::NotPositiveDefinite(negative_eigenvalue.re));
            }

            // Get inverse FFT plan
            let inverse_fft = {
                let mut planner = FFT_PLANNER.lock().map_err(|_| XError::FFTPlannerLock)?; // Lock again
                planner.plan_fft_inverse(m)
            };

            // Return computed eigenvalues and inverse plan
            (complex_data, inverse_fft)
        };

        // Generate a random Gaussian vector (components for complex noise)
        let z_real = normal::standard_rands::<f64>(m);
        let mut z_imag = normal::standard_rands::<f64>(m);

        // Special handling for real output: Z_k = conj(Z_{m-k})
        // This implies z_imag[0] = 0 and z_imag[m/2] = 0 (if m is even)
        z_imag[0] = 0.0;
        if m.is_multiple_of(2) {
            z_imag[m / 2] = 0.0;
        }

        // Build the complex vector Y = sqrt(Lambda) * Z
        // Modify 'eigenvalues' vector in place to store Y
        for i in 0..m {
            // Ensure eigenvalue is non-negative before taking sqrt
            let sqrt_lambda = eigenvalues[i].re.max(0.0).sqrt();
            // Y_k = sqrt(lambda_k) * (z_real_k + i * z_imag_k)
            eigenvalues[i] = Complex::new(sqrt_lambda * z_real[i], sqrt_lambda * z_imag[i]);
        }

        // Execute inverse FFT (modifies 'eigenvalues' vector in place to store the result)
        ifft.process(&mut eigenvalues);

        // Extract the real part of the result and normalize (IFFT result needs scaling by 1/m)
        let scale = 1.0 / m as f64;
        let result = eigenvalues
            .into_iter()
            .take(n)
            .map(|c| c.re * scale)
            .collect();

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
