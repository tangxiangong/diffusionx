//! Fractional Brownian motion simulation

use crate::{
    SimulationError, XResult,
    random::normal,
    simulation::prelude::*,
    utils::{CirculantEmbedding, cumsum, fbm_correlation},
};
use rayon::prelude::*;

/// Fractional Brownian motion
///
/// This struct represents a Fractional Brownian motion.
///
/// # Fields
///
/// * `start_position` - The starting position of the Fractional Brownian motion.
/// * `hurst_exponent` - The Hurst exponent of the Fractional Brownian motion.
#[derive(Debug, Clone)]
pub struct Fbm {
    start_position: f64,
    hurst_exponent: f64,
}

impl Fbm {
    /// Create a new Fractional Brownian motion simulation
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position of the Fractional Brownian motion.
    /// * `hurst_exponent` - The Hurst exponent of the Fractional Brownian motion.
    pub fn new(start_position: impl Into<f64>, hurst_exponent: f64) -> XResult<Self> {
        let start_position = start_position.into();
        if hurst_exponent <= 0.0 || hurst_exponent >= 1.0 {
            return Err(SimulationError::InvalidParameters(
                "hurst_exponent must be in the range (0, 1)".to_string(),
            )
            .into());
        }
        Ok(Self {
            start_position,
            hurst_exponent,
        })
    }

    /// Get the starting position of the Fractional Brownian motion simulation
    pub fn start_position(&self) -> f64 {
        self.start_position
    }

    /// Get the Hurst exponent of the Fractional Brownian motion simulation
    pub fn hurst_exponent(&self) -> f64 {
        self.hurst_exponent
    }

    /// Get the mean of the Fractional Brownian motion simulation
    ///
    /// # Returns
    ///
    /// A f64 representing the mean of the Fractional Brownian motion simulation.  
    ///
    /// # Example
    ///
    /// ```rust
    /// let fbm = Fbm::new(10.0, 1.0).unwrap();
    /// let mean = fbm.mean(1.0, 1000, 0.1).unwrap();
    /// ```
    pub fn mean(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let traj = self.duration(duration).unwrap();
        traj.raw_moment(1, particles, time_step)
    }

    /// Get the mean square displacement of the Fractional Brownian motion simulation
    ///
    /// # Returns
    ///
    /// A f64 representing the mean square displacement of the Fractional Brownian motion simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let fbm = Fbm::new(10.0, 1.0).unwrap();
    /// let msd = fbm.msd(1.0, 1000, 0.1).unwrap();
    /// ```
    pub fn msd(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let traj = self.duration(duration).unwrap();
        traj.central_moment(2, particles, time_step)
    }

    /// Get the raw moment of the Fractional Brownian motion simulation
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the Fractional Brownian motion simulation.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the Fractional Brownian motion simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the raw moment of the Fractional Brownian motion simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let fbm = Fbm::new(10.0, 1.0).unwrap();
    /// let moment = fbm.raw_moment(1.0, 1000, 0.1).unwrap();
    /// ```
    pub fn raw_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> XResult<f64> {
        let traj = self.duration(duration).unwrap();
        traj.raw_moment(order, particles, time_step)
    }

    /// Get the central moment of the Fractional Brownian motion simulation
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the Fractional Brownian motion simulation.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the Fractional Brownian motion simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the central moment of the Fractional Brownian motion simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let fbm = Fbm::new(10.0, 1.0).unwrap();
    /// let moment = fbm.central_moment(1.0, 1000, 0.1).unwrap();
    /// ```
    pub fn central_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> XResult<f64> {
        let traj = self.duration(duration).unwrap();
        traj.central_moment(order, particles, time_step)
    }

    /// Get the first passage time of the Fractional Brownian motion simulation
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the Fractional Brownian motion simulation.
    /// * `max_duration` - The maximum duration of the Fractional Brownian motion simulation.
    /// * `time_step` - The time step of the Fractional Brownian motion simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the first passage time of the Fractional Brownian motion simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let fbm = Fbm::new(10.0, 1.0).unwrap();
    /// let fpt = fbm.fpt((-1.0, 1.0), 1000.0, 0.1).unwrap();
    /// ```
    pub fn fpt(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        max_duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<Option<f64>> {
        let fpt = FirstPassageTime::new(self, domain)?;
        fpt.simulate(max_duration, time_step)
    }

    /// Get the occupation time of the Fractional Brownian motion simulation
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the Fractional Brownian motion simulation.
    /// * `duration` - The duration of the Fractional Brownian motion simulation.
    /// * `time_step` - The time step of the Fractional Brownian motion simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the occupation time of the Fractional Brownian motion simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let fbm = Fbm::new(10.0, 1.0).unwrap();
    /// let ot = fbm.occupation_time((-1.0, 1.0), 1000.0, 0.1).unwrap();
    /// ```
    pub fn occupation_time(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<f64> {
        let ot = OccupationTime::new(self, domain, duration)?;
        ot.simulate(time_step)
    }
}

/// impl `ContinuousProcess` trait for Fractional Brownian motion
impl ContinuousProcess for Fbm {
    /// Simulate Fractional Brownian motion
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the Fractional Brownian motion simulation.
    /// * `time_step` - The time step of the Fractional Brownian motion simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the Fractional Brownian motion simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let fbm = Fbm::new(10.0, 0.5).unwrap();
    /// let duration = 1.0;
    /// let time_step = 0.1;
    /// let (t, x) = fbm.simulate(duration, time_step).unwrap();
    /// ```
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        simulate_fbm(
            self.start_position,
            self.hurst_exponent,
            duration,
            time_step,
        )
    }
}

/// Simulate Fractional Brownian motion
///
/// This function simulates Fractional Brownian motion.
///
/// # Arguments
///
/// * `start_position` - The starting position of the Fractional Brownian motion.  
/// * `hurst_exponent` - The Hurst exponent of the Fractional Brownian motion.
/// * `duration` - The duration of the Fractional Brownian motion.
/// * `time_step` - The time step of the Fractional Brownian motion.
///
/// # Returns
///
/// A tuple containing the time and the position of the Fractional Brownian motion simulation.   
///
/// # Example
///
/// ```rust
/// let start_position = 10.0;
/// let hurst_exponent = 0.5;
/// let duration = 1.0;
/// let time_step = 0.1;
/// let (t, x) = simulate_fbm(start_position, hurst_exponent, duration, time_step).unwrap();
/// ```
pub fn simulate_fbm(
    start_position: impl Into<f64>,
    hurst_exponent: f64,
    duration: impl Into<f64>,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let start_position = start_position.into();
    let duration = duration.into();
    let num_steps = (duration / time_step).ceil() as usize;
    let t = (0..=num_steps)
        .into_par_iter()
        .map(|i| time_step * i as f64)
        .collect::<Vec<_>>();

    // 使用 fbm_correlation 函数创建相关函数
    let correlation_fn = fbm_correlation(hurst_exponent, time_step);

    // 创建 CirculantEmbedding 实例
    let circulant = CirculantEmbedding::new(num_steps, correlation_fn);

    // 生成噪声
    let noise = if hurst_exponent == 0.5 {
        normal::rands(0.0, 2.0 * time_step, num_steps)?
    } else {
        circulant.generate()?
    };

    // 计算累积和
    let x = cumsum(start_position, &noise);

    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::prelude::{ContinuousTrajectoryTrait, Moment};

    #[test]
    fn test_simulate_bm() {
        let fbm = Fbm::new(10.0, 0.5).unwrap();
        let duration = 1.0;
        let time_step = 0.1;
        let (t, x) = fbm.simulate(duration, time_step).unwrap();
        println!("t: {:?}", t);
        println!("x: {:?}", x);
    }

    #[test]
    fn test_raw_moment() {
        let fbm = Fbm::new(10.0, 0.5).unwrap();
        let duration = 1.0;
        let time_step = 0.1;
        let traj = fbm.duration(duration).unwrap();
        let moment = traj.raw_moment(1, 1000, time_step).unwrap();
        println!("moment: {:?}", moment);
    }

    #[test]
    fn test_fpt() {
        let fbm = Fbm::new(0.0, 0.5).unwrap();
        let time_step = 0.1;
        let fpt = fbm.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {:?}", fpt);
    }

    #[test]
    fn test_occupation_time() {
        let fbm = Fbm::new(0.0, 0.5).unwrap();
        let time_step = 0.1;
        let ot = fbm.occupation_time((-1.0, 1.0), 10.0, time_step).unwrap();
        println!("ot: {:?}", ot);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Fbm>();
    }
}
