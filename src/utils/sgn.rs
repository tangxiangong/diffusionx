use crate::{XResult, random::normal};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use realfft::{FftNum, RealFftPlanner, num_complex::Complex};
use std::ops::MulAssign;

/// Circulant embedding method for generating stationary Gaussian random fields with given correlation functions
pub struct CirculantEmbedding<F: Fn(usize) -> T, T: Float + FftNum = f64> {
    /// Number of grid points
    size: usize,
    /// Correlation function, takes distance as input and returns correlation
    correlation_fn: F,
    /// fft planner
    planner: RealFftPlanner<T>,
    /// embedding size
    embedding_size: usize,
    /// sqrt eigenvalues
    sqrt_eigenvalues: Option<Vec<T>>,
}

impl<F: Fn(usize) -> T + Send + Sync, T: Float + FftNum + Send + Sync> CirculantEmbedding<F, T> {
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
            planner: RealFftPlanner::new(),
            embedding_size: next_power_of_2(2 * size),
            sqrt_eigenvalues: None,
        }
    }

    fn embed(&mut self) -> XResult<()> {
        let mut first_row = (0..self.embedding_size)
            .into_par_iter()
            .map(|i| {
                let dist = if i < self.size {
                    i
                } else {
                    self.embedding_size - i
                };
                (self.correlation_fn)(dist)
            })
            .collect::<Vec<_>>();

        let fft = self.planner.plan_fft_forward(self.embedding_size);
        let mut eigenvalues = fft.make_output_vec();

        fft.process(&mut first_row, &mut eigenvalues)?;

        if eigenvalues.iter().any(|val| val.re < T::zero()) {
            self.embedding_size *= 2;
            self.embed()
        } else {
            self.sqrt_eigenvalues =
                Some(eigenvalues.into_iter().map(|val| val.re.sqrt()).collect());
            Ok(())
        }
    }

    /// Generate a one-dimensional stationary Gaussian random field
    pub fn generate(&mut self) -> XResult<Vec<T>>
    where
        T: FftNum + MulAssign<T>,
        StandardNormal: Distribution<T>,
    {
        if self.sqrt_eigenvalues.is_none() {
            self.embed()?;
        }

        let ifft = self.planner.plan_fft_inverse(self.embedding_size);

        let mut modified_z = self
            .sqrt_eigenvalues
            .as_ref()
            .unwrap()
            .par_iter()
            .map(|&sqrt_lambda| {
                let re = sqrt_lambda * normal::standard_rand();
                let im = sqrt_lambda * normal::standard_rand();
                Complex::new(re, im)
            })
            .collect::<Vec<_>>();

        let mut result = ifft.make_output_vec();
        ifft.process(&mut modified_z, &mut result)?;
        let scale = T::one() / T::from(self.embedding_size).unwrap();
        result.iter_mut().for_each(|x| *x *= scale);
        result.truncate(self.size);

        Ok(result)
    }
}

fn next_power_of_2(mut n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    if size_of::<usize>() == 8 {
        n |= n >> 32;
    }
    n + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_power_of_2() {
        assert_eq!(next_power_of_2(8), 8);
        assert_eq!(next_power_of_2(11), 16);
    }
}
