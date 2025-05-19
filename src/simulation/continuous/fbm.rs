//! Fractional Brownian motion simulation
//!

use crate::{
    SimulationError, XResult,
    random::normal,
    simulation::prelude::*,
    utils::{CirculantEmbedding, cumsum, fbm_correlation},
};
use rayon::prelude::*;

/// Fractional Brownian motion
#[derive(Debug, Clone)]
pub struct FBM {
    /// The starting position
    start_position: f64,
    /// The Hurst exponent
    hurst_exponent: f64,
}

impl FBM {
    /// Create a new `Fbm`
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position.
    /// * `hurst_exponent` - The Hurst exponent.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::Fbm;
    ///
    /// let fbm = Fbm::new(10.0, 0.5).unwrap();
    /// ```
    pub fn new(start_position: impl Into<f64>, hurst_exponent: f64) -> XResult<Self> {
        let start_position = start_position.into();
        if hurst_exponent <= 0.0 || hurst_exponent >= 1.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `hurst_exponent` must be in the range (0, 1), got {}",
                hurst_exponent
            ))
            .into());
        }
        Ok(Self {
            start_position,
            hurst_exponent,
        })
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> f64 {
        self.start_position
    }

    /// Get the Hurst exponent
    pub fn get_hurst_exponent(&self) -> f64 {
        self.hurst_exponent
    }
}

/// impl `ContinuousProcess` trait for `Fbm`
impl ContinuousProcess for FBM {
    /// Simulate Fractional Brownian motion
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the trajectory.
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::Fbm, prelude::*};
    ///
    /// let fbm = Fbm::new(10.0, 0.5).unwrap();
    /// let time_step = 0.1;
    /// let duration = 1.0;
    /// let (t, x) = fbm.simulate(duration, time_step).unwrap();
    /// ```
    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_fbm(
            self.start_position,
            self.hurst_exponent,
            duration,
            time_step,
        )
    }
}

/// Simulate FBM
///
/// # Arguments
///
/// * `start_position` - The starting position.
/// * `hurst_exponent` - The Hurst exponent.
/// * `duration` - The duration of the trajectory.
/// * `time_step` - The time step of the simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::fbm::simulate_fbm;
///
/// let start_position = 10.0;
/// let hurst_exponent = 0.5;
/// let duration = 1.0;
/// let time_step = 0.1;
/// let (t, x) = simulate_fbm(start_position, hurst_exponent, duration, time_step).unwrap();
/// ```
pub fn simulate_fbm(
    start_position: f64,
    hurst_exponent: f64,
    duration: f64,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let num_steps = (duration / time_step).ceil() as usize;
    let t = (0..=num_steps)
        .into_par_iter()
        .map(|i| time_step * i as f64)
        .collect::<Vec<_>>();

    // Use `fbm_correlation` function to create the correlation function
    let correlation_fn = fbm_correlation(hurst_exponent, time_step);

    // Create a `CirculantEmbedding` instance
    let circulant = CirculantEmbedding::new(num_steps, correlation_fn);

    // Generate fBn
    let noise = if hurst_exponent == 0.5 {
        normal::rands(0.0, 2.0 * time_step, num_steps)?
    } else {
        circulant.generate()?
    };

    // Calculate the cumulative sum
    let x = cumsum(start_position, &noise);

    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::prelude::Moment;

    #[test]
    fn test_simulate_bm() {
        let fbm = FBM::new(10.0, 0.5).unwrap();
        let duration = 1.0;
        let time_step = 0.1;
        let (t, x) = fbm.simulate(duration, time_step).unwrap();
        println!("t: {:?}", t);
        println!("x: {:?}", x);
    }

    #[test]
    fn test_raw_moment() {
        let fbm = FBM::new(10.0, 0.5).unwrap();
        let duration = 1.0;
        let time_step = 0.1;
        let traj = fbm.duration(duration).unwrap();
        let moment = traj.raw_moment(1, 1000, time_step).unwrap();
        println!("moment: {:?}", moment);
    }

    #[test]
    fn test_fpt() {
        let fbm = FBM::new(0.0, 0.5).unwrap();
        let time_step = 0.1;
        let fpt = fbm.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {:?}", fpt);
    }

    #[test]
    fn test_occupation_time() {
        let fbm = FBM::new(0.0, 0.5).unwrap();
        let time_step = 0.1;
        let ot = fbm.occupation_time((-1.0, 1.0), 10.0, time_step).unwrap();
        println!("ot: {:?}", ot);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FBM>();
    }
}
