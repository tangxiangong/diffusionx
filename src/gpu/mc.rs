//! GPU Monte Carlo simulation module
//!
//! This module provides true GPU-accelerated Monte Carlo simulation where
//! statistics are computed directly on GPU without transferring all trajectories.

use crate::{XError, XResult};

/// Monte Carlo statistics computed on GPU
#[derive(Debug, Clone)]
pub struct MonteCarloStats {
    /// Time points
    pub times: Vec<f64>,

    /// Mean position at each time
    pub mean: Vec<f64>,

    /// Mean square displacement at each time
    pub msd: Vec<f64>,

    /// Variance at each time
    pub variance: Vec<f64>,

    /// Number of particles simulated
    pub num_particles: usize,
}

impl MonteCarloStats {
    /// Create new Monte Carlo statistics
    pub fn new(num_steps: usize, num_particles: usize) -> Self {
        Self {
            times: Vec::with_capacity(num_steps + 1),
            mean: vec![0.0; num_steps + 1],
            msd: vec![0.0; num_steps + 1],
            variance: vec![0.0; num_steps + 1],
            num_particles,
        }
    }

    /// Get standard deviation at each time
    pub fn std_dev(&self) -> Vec<f64> {
        self.variance.iter().map(|v| v.sqrt()).collect()
    }

    /// Get standard error at each time
    pub fn standard_error(&self) -> Vec<f64> {
        let n = self.num_particles as f64;
        self.variance.iter().map(|v| (v / n).sqrt()).collect()
    }
}

/// Monte Carlo moment statistics
#[derive(Debug, Clone)]
pub struct MomentStats {
    /// Time points
    pub times: Vec<f64>,

    /// Raw moments at each time [order][time]
    pub raw_moments: Vec<Vec<f64>>,

    /// Central moments at each time [order][time]
    pub central_moments: Vec<Vec<f64>>,

    /// Number of particles
    pub num_particles: usize,
}

impl MomentStats {
    /// Create new moment statistics
    pub fn new(max_order: usize, num_steps: usize, num_particles: usize) -> Self {
        Self {
            times: Vec::with_capacity(num_steps + 1),
            raw_moments: vec![vec![0.0; num_steps + 1]; max_order],
            central_moments: vec![vec![0.0; num_steps + 1]; max_order],
            num_particles,
        }
    }

    /// Get skewness at each time (normalized 3rd central moment)
    pub fn skewness(&self) -> Option<Vec<f64>> {
        if self.central_moments.len() < 3 {
            return None;
        }

        let m3 = &self.central_moments[2]; // 3rd moment (0-indexed)
        let variance = &self.central_moments[1]; // 2nd moment

        Some(
            m3.iter()
                .zip(variance.iter())
                .map(|(m3, v)| if *v > 0.0 { m3 / v.powf(1.5) } else { 0.0 })
                .collect(),
        )
    }

    /// Get kurtosis at each time (normalized 4th central moment)
    pub fn kurtosis(&self) -> Option<Vec<f64>> {
        if self.central_moments.len() < 4 {
            return None;
        }

        let m4 = &self.central_moments[3]; // 4th moment
        let variance = &self.central_moments[1]; // 2nd moment

        Some(
            m4.iter()
                .zip(variance.iter())
                .map(|(m4, v)| if *v > 0.0 { m4 / (v * v) } else { 0.0 })
                .collect(),
        )
    }
}

/// Configuration for Monte Carlo simulation
#[derive(Debug, Clone)]
pub struct MonteCarloConfig {
    /// Number of Monte Carlo samples (particles)
    pub num_samples: usize,

    /// Whether to compute raw moments
    pub compute_raw_moments: bool,

    /// Whether to compute central moments
    pub compute_central_moments: bool,

    /// Maximum moment order to compute
    pub max_moment_order: usize,

    /// Whether to compute autocorrelation
    pub compute_autocorrelation: bool,

    /// Autocorrelation lag steps
    pub autocorrelation_lags: usize,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            num_samples: 10000,
            compute_raw_moments: false,
            compute_central_moments: false,
            max_moment_order: 2,
            compute_autocorrelation: false,
            autocorrelation_lags: 100,
        }
    }
}

impl MonteCarloConfig {
    /// Create new Monte Carlo configuration
    pub fn new(num_samples: usize) -> Self {
        Self {
            num_samples,
            ..Default::default()
        }
    }

    /// Enable moment computation
    pub fn with_moments(mut self, max_order: usize) -> Self {
        self.compute_raw_moments = true;
        self.compute_central_moments = true;
        self.max_moment_order = max_order;
        self
    }

    /// Enable autocorrelation computation
    pub fn with_autocorrelation(mut self, lags: usize) -> Self {
        self.compute_autocorrelation = true;
        self.autocorrelation_lags = lags;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> XResult<()> {
        if self.num_samples == 0 {
            return Err(XError::InvalidParameters(
                "num_samples must be greater than 0".to_string(),
            ));
        }

        if self.max_moment_order == 0 {
            return Err(XError::InvalidParameters(
                "max_moment_order must be greater than 0".to_string(),
            ));
        }

        if self.max_moment_order > 10 {
            return Err(XError::InvalidParameters(
                "max_moment_order should not exceed 10".to_string(),
            ));
        }

        Ok(())
    }
}

/// First passage time statistics
#[derive(Debug, Clone)]
pub struct FirstPassageStats {
    /// Mean first passage time
    pub mean_fpt: f64,

    /// Standard deviation of FPT
    pub std_fpt: f64,

    /// Fraction of particles that crossed the boundary
    pub crossing_fraction: f64,

    /// Number of samples
    pub num_samples: usize,
}

impl FirstPassageStats {
    /// Create new FPT statistics
    pub fn from_times(times: &[Option<f64>]) -> Self {
        let num_samples = times.len();
        let crossing_times: Vec<f64> = times.iter().filter_map(|&t| t).collect();
        let num_crossings = crossing_times.len();

        if num_crossings == 0 {
            return Self {
                mean_fpt: f64::INFINITY,
                std_fpt: 0.0,
                crossing_fraction: 0.0,
                num_samples,
            };
        }

        let mean = crossing_times.iter().sum::<f64>() / num_crossings as f64;
        let variance = crossing_times
            .iter()
            .map(|t| (t - mean).powi(2))
            .sum::<f64>()
            / num_crossings as f64;

        Self {
            mean_fpt: mean,
            std_fpt: variance.sqrt(),
            crossing_fraction: num_crossings as f64 / num_samples as f64,
            num_samples,
        }
    }
}

/// Probability distribution estimated from Monte Carlo
#[derive(Debug, Clone)]
pub struct ProbabilityDistribution {
    /// Bin centers
    pub bins: Vec<f64>,

    /// Probability density at each bin
    pub density: Vec<f64>,

    /// Cumulative distribution
    pub cumulative: Vec<f64>,
}

impl ProbabilityDistribution {
    /// Create histogram from data
    pub fn from_samples(data: &[f64], num_bins: usize) -> Self {
        if data.is_empty() {
            return Self {
                bins: Vec::new(),
                density: Vec::new(),
                cumulative: Vec::new(),
            };
        }

        let min = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        if range == 0.0 {
            return Self {
                bins: vec![min],
                density: vec![1.0],
                cumulative: vec![1.0],
            };
        }

        let bin_width = range / num_bins as f64;
        let mut counts = vec![0usize; num_bins];

        for &value in data {
            let bin_idx = ((value - min) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(num_bins - 1);
            counts[bin_idx] += 1;
        }

        let bins: Vec<f64> = (0..num_bins)
            .map(|i| min + (i as f64 + 0.5) * bin_width)
            .collect();

        let total = data.len() as f64;
        let density: Vec<f64> = counts
            .iter()
            .map(|&c| c as f64 / (total * bin_width))
            .collect();

        let mut cumulative = Vec::with_capacity(num_bins);
        let mut cum_sum = 0.0;
        for &d in &density {
            cum_sum += d * bin_width;
            cumulative.push(cum_sum);
        }

        Self {
            bins,
            density,
            cumulative,
        }
    }

    /// Find percentile
    pub fn percentile(&self, p: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&p) {
            return None;
        }

        for (i, &cum) in self.cumulative.iter().enumerate() {
            if cum >= p {
                return Some(self.bins[i]);
            }
        }

        self.bins.last().copied()
    }

    /// Get median
    pub fn median(&self) -> Option<f64> {
        self.percentile(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_montecarlo_stats() {
        let stats = MonteCarloStats::new(100, 1000);
        assert_eq!(stats.mean.len(), 101);
        assert_eq!(stats.msd.len(), 101);
        assert_eq!(stats.variance.len(), 101);
        assert_eq!(stats.num_particles, 1000);

        let std_dev = stats.std_dev();
        assert_eq!(std_dev.len(), 101);
    }

    #[test]
    fn test_montecarlo_config() {
        let config = MonteCarloConfig::new(5000)
            .with_moments(4)
            .with_autocorrelation(50);

        assert_eq!(config.num_samples, 5000);
        assert!(config.compute_raw_moments);
        assert!(config.compute_central_moments);
        assert_eq!(config.max_moment_order, 4);
        assert!(config.compute_autocorrelation);
        assert_eq!(config.autocorrelation_lags, 50);

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_first_passage_stats() {
        let times = vec![Some(1.0), Some(2.0), None, Some(1.5), None];
        let stats = FirstPassageStats::from_times(&times);

        assert_eq!(stats.num_samples, 5);
        assert_eq!(stats.crossing_fraction, 0.6);
        assert!((stats.mean_fpt - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_probability_distribution() {
        let data = vec![1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let dist = ProbabilityDistribution::from_samples(&data, 4);

        assert_eq!(dist.bins.len(), 4);
        assert_eq!(dist.density.len(), 4);
        assert_eq!(dist.cumulative.len(), 4);

        let median = dist.median();
        assert!(median.is_some());
    }

    #[test]
    fn test_moment_stats() {
        let stats = MomentStats::new(4, 100, 1000);
        assert_eq!(stats.raw_moments.len(), 4);
        assert_eq!(stats.central_moments.len(), 4);
        assert_eq!(stats.raw_moments[0].len(), 101);
    }
}
