//! Brownian excursion simulation
//!
//! For Brownian motion and Brownian bridge, see [`crate::simulation::continuous::bm`] and [`crate::simulation::continuous::brownian_bridge`], respectively.

use crate::{SimulationError, XResult, simulation::prelude::*, utils::minmax};
use rayon::prelude::*;

use super::BrownianBridge;

/// Brownian excursion
///
/// This struct represents a Brownian excursion.
#[derive(Debug, Clone)]
pub struct BrownianExcursion;

impl BrownianExcursion {
    /// Get the mean of the Brownian excursion simulation
    ///
    /// # Returns
    ///
    /// A f64 representing the mean of the Brownian excursion simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::BrownianExcursion, prelude::*};
    /// let be = BrownianExcursion;
    /// let mean = be.mean(1.0, 1000, 0.1).unwrap();
    /// ```
    pub fn mean(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let duration: f64 = duration.into();
        if duration > 1.0 {
            return Err(SimulationError::InvalidParameters(
                "duration must be less than or equal to 1.0".to_string(),
            )
            .into());
        }
        let traj = self.duration(duration)?;
        traj.raw_moment(1, particles, time_step)
    }
    /// Get the mean square displacement of the Brownian excursion simulation
    ///
    /// # Returns
    ///
    /// A f64 representing the mean square displacement of the Brownian excursion simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::BrownianExcursion, prelude::*};
    /// let be = BrownianExcursion;
    /// let msd = be.msd(1.0, 1000, 0.1).unwrap();
    /// ```
    pub fn msd(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let duration: f64 = duration.into();
        if duration > 1.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be less than or equal to 1.0, got {}",
                duration
            ))
            .into());
        }
        let traj = self.duration(duration)?;
        traj.central_moment(2, particles, time_step)
    }

    /// Get the raw moment of the Brownian bridge simulation
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the Brownian excursion simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the raw moment of the Brownian excursion simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::BrownianExcursion, prelude::*};
    /// let be = BrownianExcursion;
    /// let moment = be.raw_moment(1.0, 1000, 0.1).unwrap();
    /// ```
    pub fn raw_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> XResult<f64> {
        let duration: f64 = duration.into();
        if duration > 1.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be less than or equal to 1.0, got {}",
                duration
            ))
            .into());
        }
        let traj = self.duration(duration)?;
        traj.raw_moment(order, particles, time_step)
    }

    /// Get the central moment of the Brownian excursion simulation
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the Brownian excursion simulation.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the Brownian excursion simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the central moment of the Brownian excursion simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::BrownianExcursion, prelude::*};
    /// let be = BrownianExcursion;
    /// let msd = bb.msd(1.0, 1000, 0.1).unwrap();
    /// ```
    pub fn central_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> XResult<f64> {
        let duration: f64 = duration.into();
        if duration > 1.0 {
            return Err(SimulationError::InvalidParameters(
                "duration must be less than or equal to 1.0".to_string(),
            )
            .into());
        }
        let traj = self.duration(duration)?;
        traj.central_moment(order, particles, time_step)
    }

    /// Get the first passage time of the Brownian excursion simulation
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the Brownian excursion simulation.
    /// * `time_step` - The time step of the Brownian excursion simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the first passage time of the Brownian excursion simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::BrownianExcursion, prelude::*};
    /// let be = BrownianExcursion;
    /// let fpt = be.fpt((-1.0, 1.0), 0.1).unwrap();
    /// ```
    pub fn fpt(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        time_step: f64,
    ) -> XResult<Option<f64>> {
        if time_step <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `time_step` must be positive, got {}",
                time_step
            ))
            .into());
        }
        let a: f64 = domain.0.into();
        let b: f64 = domain.1.into();

        let (t, x) = self.simulate(1, time_step)?;
        if let Some(index) = x.iter().position(|&x| x <= a || x >= b) {
            Ok(Some(t[index]))
        } else {
            Ok(None)
        }
    }

    /// Get the occupation time of the Brownian excursion simulation
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the Brownian excursion simulation.
    /// * `duration` - The duration of the Brownian excursion simulation.
    /// * `time_step` - The time step of the Brownian excursion simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the occupation time of the Brownian excursion simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::BrownianExcursion, prelude::*};
    /// let be = BrownianExcursion;
    /// let ot = be.occupation_time((-1.0, 1.0), 1000.0, 0.1).unwrap();
    /// ```
    pub fn occupation_time(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<f64> {
        let duration: f64 = duration.into();
        if duration > 1.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be less than or equal to 1.0, got {}",
                duration
            ))
            .into());
        }
        let ot = OccupationTime::new(self, domain, duration)?;
        ot.simulate(time_step)
    }
}

/// impl `ContinuousProcess` trait for Brownian excursion
impl ContinuousProcess for BrownianExcursion {
    /// Simulate Brownian excursion
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the Brownian excursion simulation.
    /// * `time_step` - The time step of the Brownian excursion simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the Brownian excursion simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::BrownianExcursion, prelude::*};
    /// let be = BrownianExcursion;
    /// let time_step = 0.1;
    /// let duration = 1.0;
    /// let (t, x) = be.simulate(duration, time_step).unwrap();
    /// ```
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        simulate_brownian_excursion(duration.into(), time_step)
    }
}

/// Simulate Brownian excursion
///
/// This function simulates Brownian excursion.
///
/// # Arguments
///
/// * `duration` - The duration of the Brownian excursion.
/// * `time_step` - The time step of the Brownian excursion.
///
/// # Returns
///
/// A tuple containing the time and the position of the Brownian excursion simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::brownian_excursion::simulate_brownian_excursion;
/// let time_step = 0.1;
/// let duration = 1.0;
/// let (t, x) = simulate_brownian_excursion(duration, time_step).unwrap();
/// ```
pub fn simulate_brownian_excursion(
    duration: impl Into<f64>,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let duration: f64 = duration.into();
    if duration <= 0.0 || duration > 1.0 {
        // Duration must be positive and not exceed 1.0
        return Err(SimulationError::InvalidParameters(format!(
            "The `duration` must be in (0.0, 1.0], got {}",
            duration
        ))
        .into());
    }
    if time_step <= 0.0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `time_step` must be positive, got {}",
            time_step
        ))
        .into());
    }

    let bridge = BrownianBridge;
    let (bridge_t, bridge_traj) = bridge.simulate(1.0, time_step)?;

    // Find the minimum value and its index within [0, 1]
    let (min_traj, _) = minmax(&bridge_traj); // Assuming minmax returns Result

    let min_traj_index = bridge_traj
        .iter()
        .position(|&x| (x - min_traj).abs() < f64::EPSILON) // Use tolerance for float comparison
        .ok_or(SimulationError::Unknown)?;

    let tau_m = bridge_t[min_traj_index];

    let x = bridge_t
        .par_iter() // Parallelize the mapping
        .map(|t_i| {
            // Correct modulo calculation for tt
            let tt = (t_i + tau_m) % 1.0;

            // Find the index in the original bridge time corresponding to tt
            // Use >= for better accuracy and handle potential errors
            let index = bridge_t.iter().position(|&t| t >= tt).unwrap();

            // Apply the transformation
            bridge_traj[index] - min_traj
        })
        .collect(); // Collect results from parallel iterator

    Ok((bridge_t, x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::prelude::{ContinuousTrajectoryTrait, Moment};

    #[test]
    fn test_simulate_be() {
        let be = BrownianExcursion;
        let duration = 1.0;
        let time_step = 0.1;
        let (t, x) = be.simulate(duration, time_step).unwrap();
        println!("t: {:?}", t);
        println!("x: {:?}", x);
    }

    #[test]
    fn test_raw_moment() {
        let be = BrownianExcursion;
        let duration = 1.0;
        let time_step = 0.1;
        let traj = be.duration(duration).unwrap();
        let moment = traj.raw_moment(1, 1000, time_step).unwrap();
        println!("moment: {:?}", moment);
    }

    #[test]
    fn test_fpt() {
        let be = BrownianExcursion;
        let time_step = 0.1;
        let fpt = be.fpt((-1.0, 1.0), time_step).unwrap();
        println!("fpt: {:?}", fpt);
    }

    #[test]
    fn test_occupation_time() {
        let be = BrownianExcursion;
        let time_step = 0.1;
        let ot = be.occupation_time((-1.0, 1.0), 1.0, time_step).unwrap();
        println!("ot: {:?}", ot);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BrownianExcursion>();
    }
}
