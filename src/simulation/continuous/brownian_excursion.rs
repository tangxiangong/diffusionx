//! Brownian excursion simulation

use crate::{
    SimulationError, XResult,
    simulation::{continuous::BrownianBridge, prelude::*},
    utils::minmax,
};
use rayon::prelude::*;

/// Brownian excursion
#[derive(Debug, Clone)]
pub struct BrownianExcursion;

impl BrownianExcursion {
    /// Get the first passage time of the Brownian excursion simulation
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the Brownian excursion simulation.
    /// * `time_step` - The time step of the Brownian excursion simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::BrownianExcursion;
    ///
    /// let be = BrownianExcursion;
    /// let fpt = be.fpt((-1.0, 1.0), 0.1).unwrap();
    /// ```
    pub fn fpt(&self, domain: (f64, f64), time_step: f64) -> XResult<Option<f64>> {
        let (a, b) = domain;
        let (t, x) = self.simulate_unchecked(1.0, time_step)?;
        if let Some(index) = x.iter().position(|&x| x <= a || x >= b) {
            Ok(Some(t[index]))
        } else {
            Ok(None)
        }
    }
}

impl ContinuousProcess for BrownianExcursion {
    fn start(&self) -> f64 {
        0.0
    }

    fn simulate_unchecked(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_brownian_excursion(duration, time_step)
    }

    fn displacement(&self, _: f64, _: f64) -> XResult<f64> {
        Ok(0.0)
    }
}

/// Simulate Brownian excursion
///
/// # Arguments
///
/// * `duration` - The duration of the trajectory.
/// * `time_step` - The time step of the simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::brownian_excursion::simulate_brownian_excursion;
///
/// let time_step = 0.1;
/// let duration = 1.0;
/// let (t, x) = simulate_brownian_excursion(duration, time_step).unwrap();
/// ```
pub fn simulate_brownian_excursion(duration: f64, time_step: f64) -> XResult<(Vec<f64>, Vec<f64>)> {
    if duration > 1.0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `duration` must be in (0.0, 1.0], got {duration}"
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
    use crate::simulation::prelude::Moment;

    #[test]
    fn test_simulate_be() {
        let be = BrownianExcursion;
        let duration = 1.0;
        let time_step = 0.1;
        let (t, x) = be.simulate(duration, time_step).unwrap();
        println!("t: {t:?}");
        println!("x: {x:?}");
    }

    #[test]
    fn test_raw_moment() {
        let be = BrownianExcursion;
        let duration = 1.0;
        let time_step = 0.1;
        let traj = be.duration(duration).unwrap();
        let moment = traj.raw_moment(1, 1000, time_step).unwrap();
        println!("moment: {moment:?}");
    }

    #[test]
    fn test_fpt() {
        let be = BrownianExcursion;
        let time_step = 0.1;
        let fpt = be.fpt((-1.0, 1.0), time_step).unwrap();
        println!("fpt: {fpt:?}");
    }

    #[test]
    fn test_occupation_time() {
        let be = BrownianExcursion;
        let time_step = 0.1;
        let ot = be.occupation_time((-1.0, 1.0), 1.0, time_step).unwrap();
        println!("ot: {ot:?}");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BrownianExcursion>();
    }
}
