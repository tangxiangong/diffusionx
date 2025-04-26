//! Brownian meander simulation
//!
//! For Brownian motion, Brownian bridge, Brownian excursion, see [`crate::simulation::continuous::bm`], [`crate::simulation::continuous::brownian_bridge`], and [`crate::simulation::continuous::brownian_excursion`], respectively.

use crate::{SimulationError, XResult, simulation::prelude::*, utils::float_eq};
use rayon::prelude::*;

use super::Bm;

/// Brownian meander
///
/// This struct represents a Brownian meander.
#[derive(Debug, Clone)]
pub struct BrownianMeander;

impl BrownianMeander {
    /// Get the mean of the Brownian meander simulation
    ///
    /// # Returns
    ///
    /// A f64 representing the mean of the Brownian meander simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::BrownianMeander, prelude::*};
    /// let bm = BrownianMeander;
    /// let mean = bm.mean(1.0, 1000, 0.1).unwrap();
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
    /// Get the mean square displacement of the Brownian meander simulation
    ///
    /// # Returns
    ///
    /// A f64 representing the mean square displacement of the Brownian meander simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::BrownianMeander, prelude::*};
    /// let bm = BrownianMeander;
    /// let msd = bm.msd(1.0, 1000, 0.1).unwrap();
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

    /// Get the raw moment of the Brownian meander simulation
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the Brownian meander simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the raw moment of the Brownian meander simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::BrownianMeander, prelude::*};
    /// let bm = BrownianMeander;
    /// let moment = bm.raw_moment(1.0, 1000, 0.1).unwrap();
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

    /// Get the central moment of the Brownian meander simulation
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the Brownian meander simulation.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    /// * `time_step` - The time step of the Brownian meander simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the central moment of the Brownian meander simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::BrownianMeander, prelude::*};
    /// let bm = BrownianMeander;
    /// let msd = bm.msd(1.0, 1000, 0.1).unwrap();
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

    /// Get the first passage time of the Brownian meander simulation
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the Brownian meander simulation.
    /// * `time_step` - The time step of the Brownian meander simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the first passage time of the Brownian meander simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::BrownianMeander, prelude::*};
    /// let bm = BrownianMeander;
    /// let fpt = bm.fpt((-1.0, 1.0), 0.1).unwrap();
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

    /// Get the occupation time of the Brownian meander simulation
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the Brownian meander simulation.
    /// * `duration` - The duration of the Brownian meander simulation.
    /// * `time_step` - The time step of the Brownian meander simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the occupation time of the Brownian meander simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::BrownianMeander, prelude::*};
    /// let bm = BrownianMeander;
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

/// impl `ContinuousProcess` trait for Brownian meander
impl ContinuousProcess for BrownianMeander {
    /// Simulate Brownian meander
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the Brownian meander simulation.
    /// * `time_step` - The time step of the Brownian meander simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::BrownianMeander, prelude::*};
    /// let bm = BrownianMeander;
    /// let time_step = 0.1;
    /// let duration = 1.0;
    /// let (t, x) = be.simulate(duration, time_step).unwrap();
    /// ```
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        simulate_brownian_meander(duration.into(), time_step)
    }
}

/// Simulate Brownian meander
///
/// This function simulates Brownian meander.
///
/// # Arguments
///
/// * `duration` - The duration of the Brownian meander.
/// * `time_step` - The time step of the Brownian meander.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::brownian_meander::simulate_brownian_meander;
/// let time_step = 0.1;
/// let duration = 1.0;
/// let (t, x) = simulate_brownian_meander(duration, time_step).unwrap();
/// ```
pub fn simulate_brownian_meander(
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

    let bm = Bm::default();
    let (bm_t, bm_traj) = bm.simulate(1.0, time_step)?;

    let hint_indexes = bm_traj
        .iter()
        .enumerate()
        .filter(|(_, x)| float_eq(**x, 0.0))
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    let last_hint_index = *hint_indexes.last().unwrap_or(&0);
    let tau = if last_hint_index == bm_t.len() - 1 {
        1.0 - time_step
    } else {
        bm_t[last_hint_index]
    };
    let coe = 1.0 / (1.0 - tau).sqrt();
    let x = bm_t
        .par_iter()
        .map(|&t_i| {
            let time = t_i * (1.0 - tau) + tau;
            let right_index = bm_t
                .iter()
                .position(|&t| t > time)
                .unwrap_or(bm_t.len() - 1);
            let left_index = right_index - 1;
            let left_time = bm_t[left_index];
            let right_time = bm_t[right_index];
            let left_value = bm_traj[left_index];
            let right_value = bm_traj[right_index];
            let k = (left_value - right_value) / (left_time - right_time);
            let value = k * (time - left_time) + left_value;
            value.abs() * coe
        })
        .collect();
    Ok((bm_t, x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::prelude::{ContinuousTrajectoryTrait, Moment};

    #[test]
    fn test_simulate_bm() {
        let bm = BrownianMeander;
        let duration = 1.0;
        let time_step = 0.1;
        let (t, x) = bm.simulate(duration, time_step).unwrap();
        println!("t: {:?}", t);
        println!("x: {:?}", x);
    }

    #[test]
    fn test_raw_moment() {
        let bm = BrownianMeander;
        let duration = 1.0;
        let time_step = 0.1;
        let traj = bm.duration(duration).unwrap();
        let moment = traj.raw_moment(1, 1000, time_step).unwrap();
        println!("moment: {:?}", moment);
    }

    #[test]
    fn test_fpt() {
        let bm = BrownianMeander;
        let time_step = 0.1;
        let fpt = bm.fpt((-1.0, 1.0), time_step).unwrap();
        println!("fpt: {:?}", fpt);
    }

    #[test]
    fn test_occupation_time() {
        let bm = BrownianMeander;
        let time_step = 0.1;
        let ot = bm.occupation_time((-1.0, 1.0), 1.0, time_step).unwrap();
        println!("ot: {:?}", ot);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BrownianMeander>();
    }
}
