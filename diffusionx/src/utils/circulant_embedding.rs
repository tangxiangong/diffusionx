use crate::random::normal;
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};

/// Circulant embedding method for generating stationary Gaussian random fields with given correlation functions
///
/// # Fields
///
/// * `size` - Number of grid points per dimension
/// * `correlation_fn` - Correlation function, takes distance as input and returns correlation
pub struct CirculantEmbedding {
    size: usize,
    correlation_fn: Box<dyn Fn(f64) -> f64 + Send + Sync + 'static>,
}

impl CirculantEmbedding {
    /// Create a new one-dimensional circulant embedding instance
    ///
    /// # Parameters
    ///
    /// * `size` - Number of grid points per dimension
    /// * `correlation_fn` - Correlation function, takes distance as input and returns correlation
    pub fn new<F>(size: usize, correlation_fn: F) -> Self
    where
        F: Fn(f64) -> f64 + Send + Sync + 'static,
    {
        CirculantEmbedding {
            size,
            correlation_fn: Box::new(correlation_fn),
        }
    }

    /// Generate a one-dimensional stationary Gaussian random field
    pub fn generate(&self) -> Vec<f64> {
        let n = self.size;
        let m = 2 * n;

        // Build the first row of the circulant embedding matrix (values of the correlation function at different distances)
        let first_row = (0..m)
            .into_par_iter()
            .map(|i| {
                let dist = if i <= m / 2 { i as f64 } else { (m - i) as f64 };
                (self.correlation_fn)(dist)
            })
            .collect::<Vec<_>>();

        // Calculate the eigenvalues (using FFT)
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(m);

        let mut complex_data: Vec<Complex<f64>> = first_row
            .into_par_iter()
            .map(|x| Complex::new(x, 0.0))
            .collect();

        fft.process(&mut complex_data);

        // Check if all eigenvalues are positive
        for val in &complex_data {
            if val.re < -1e-10 {
                panic!(
                    "Circulant embedding matrix is not positive definite, eigenvalue: {}",
                    val.re
                );
            }
        }

        // Generate a random Gaussian vector
        let z_real = normal::standard_rands(m);
        let mut z_imag = normal::standard_rands(m);

        // Special handling to ensure real output
        z_imag[0] = 0.0;
        if m % 2 == 0 {
            z_imag[m / 2] = 0.0;
        }

        // Build the complex vector and multiply by the square root of the eigenvalues
        let mut complex_result = (0..m)
            .into_par_iter()
            .map(|i| {
                let sqrt_lambda = complex_data[i].re.max(0.0).sqrt();
                let real_part = sqrt_lambda * z_real[i];
                let imag_part = sqrt_lambda * z_imag[i];
                Complex::new(real_part, imag_part)
            })
            .collect::<Vec<_>>();

        // Perform inverse FFT
        let ifft = planner.plan_fft_inverse(m);
        ifft.process(&mut complex_result);

        // Extract the result and scale
        complex_result
            .into_iter()
            .take(n)
            .map(|c| c.re)
            .collect::<Vec<_>>()

        // // 归一化处理，确保方差为1
        // let mean = result.iter().sum::<f64>() / n as f64;
        // let variance = result.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        // let scale_factor = 1.0 / variance.sqrt();

        // for i in 0..n {
        //     result[i] = (result[i] - mean) * scale_factor;
        // }

        // result
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
    move |r: f64| (-(r / length_scale).powi(2)).exp()
}

/// Matérn correlation function (nu=1/2): exp(-r/l)
pub fn matern_half_correlation(length_scale: f64) -> impl Fn(f64) -> f64 {
    move |r: f64| (-r / length_scale).exp()
}

/// Matérn correlation function (nu=3/2): (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)
pub fn matern_three_half_correlation(length_scale: f64) -> impl Fn(f64) -> f64 {
    move |r: f64| {
        let scaled_r = 3.0_f64.sqrt() * r / length_scale;
        (1.0 + scaled_r) * (-scaled_r).exp()
    }
}

/// Matérn correlation function (nu=5/2): (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
pub fn matern_five_half_correlation(length_scale: f64) -> impl Fn(f64) -> f64 {
    move |r: f64| {
        let scaled_r = 5.0_f64.sqrt() * r / length_scale;
        (1.0 + scaled_r + scaled_r.powi(2) / 3.0) * (-scaled_r).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::E;
    #[test]
    fn test_gaussian_field() {
        let size = 64;
        let length_scale = 10.0;
        let corr_fn = gaussian_correlation(length_scale);
        let embedding = CirculantEmbedding::new(size, corr_fn);

        let field = embedding.generate();

        // Check the size of the generated field
        assert_eq!(field.len(), size);

        // Calculate the sample mean (should be close to 0)
        let mean = field.iter().sum::<f64>() / size as f64;
        assert!(mean.abs() < 0.5); // Allow some statistical fluctuations

        // Calculate the sample variance (should be close to 1)
        let variance = field.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / size as f64;
        assert!((variance - 1.0).abs() < 0.5); // Allow some statistical fluctuations
    }

    #[test]
    fn test_correlation_functions() {
        let length_scale = 2.0;

        // Test the exponential correlation function
        let exp_corr = exponential_correlation(length_scale);
        assert_eq!(exp_corr(0.0), 1.0);
        assert_eq!(exp_corr(length_scale), 1.0 / E);

        // Test the Gaussian correlation function
        let gauss_corr = gaussian_correlation(length_scale);
        assert_eq!(gauss_corr(0.0), 1.0);
        assert_eq!(gauss_corr(length_scale), (-1.0_f64).exp());
    }
}
