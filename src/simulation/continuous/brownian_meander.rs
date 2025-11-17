//! Brownian meander simulation

use crate::{
    SimulationError, XResult,
    simulation::{continuous::Bm, prelude::*},
    utils::float_eq,
};
use rayon::prelude::*;

/// Brownian meander
#[derive(Debug, Clone)]
pub struct BrownianMeander;

impl BrownianMeander {
    /// Get the first passage time of the Brownian meander simulation
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the Brownian meander simulation.
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::BrownianMeander;
    ///
    /// let bm = BrownianMeander;
    /// let fpt = bm.fpt((-1.0, 1.0), 0.1).unwrap();
    /// ```
    pub fn fpt(&self, domain: (f64, f64), time_step: f64) -> XResult<Option<f64>> {
        if domain.0 >= domain.1 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `domain` must be in (a, b), got `{}` >= `{}`",
                domain.0, domain.1
            ))
            .into());
        }
        let (a, b) = domain;

        let (t, x) = self.simulate(1.0, time_step)?;
        if let Some(index) = x.iter().position(|&x| x <= a || x >= b) {
            Ok(Some(t[index]))
        } else {
            Ok(None)
        }
    }
}

impl ContinuousProcess for BrownianMeander {
    fn start(&self) -> f64 {
        0.0
    }

    fn simulate_unchecked(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_brownian_meander(duration, time_step)
    }

    fn displacement(&self, duration: f64, time_step: f64) -> XResult<f64> {
        if duration > 1.0 {
            // Duration must be positive and not exceed 1.0
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be in (0.0, 1.0], got {duration}"
            ))
            .into());
        }

        let bm = Bm::default();
        let (bm_t, bm_traj) = bm.simulate_unchecked(1.0, time_step)?;

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
        Ok(bm_t
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
            .sum())
    }
}

/// Simulate Brownian meander
///
/// # Arguments
///
/// * `duration` - The duration of the trajectory.
/// * `time_step` - The time step of the simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::brownian_meander::simulate_brownian_meander;
///
/// let time_step = 0.1;
/// let duration = 1.0;
/// let (t, x) = simulate_brownian_meander(duration, time_step).unwrap();
/// ```
pub fn simulate_brownian_meander(duration: f64, time_step: f64) -> XResult<(Vec<f64>, Vec<f64>)> {
    if duration > 1.0 {
        // Duration must be positive and not exceed 1.0
        return Err(SimulationError::InvalidParameters(format!(
            "The `duration` must be in (0.0, 1.0], got {duration}"
        ))
        .into());
    }

    let bm = Bm::default();
    let (bm_t, bm_traj) = bm.simulate_unchecked(1.0, time_step)?;

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
    use crate::simulation::prelude::Moment;

    #[test]
    fn test_simulate_meander() {
        let bm = BrownianMeander;
        let duration = 1.0;
        let time_step = 0.1;
        let (t, x) = bm.simulate(duration, time_step).unwrap();
        println!("t: {t:?}");
        println!("x: {x:?}");
    }

    #[test]
    fn test_raw_moment() {
        let bm = BrownianMeander;
        let duration = 1.0;
        let time_step = 0.1;
        let traj = bm.duration(duration).unwrap();
        let moment = traj.raw_moment(1, 1000, time_step).unwrap();
        println!("moment: {moment:?}");
    }

    #[test]
    fn test_fpt() {
        let bm = BrownianMeander;
        let time_step = 0.1;
        let fpt = bm.fpt((-1.0, 1.0), time_step).unwrap();
        println!("fpt: {fpt:?}");
    }

    #[test]
    fn test_occupation_time() {
        let bm = BrownianMeander;
        let time_step = 0.1;
        let ot = bm.occupation_time((-1.0, 1.0), 1.0, time_step).unwrap();
        println!("ot: {ot:?}");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BrownianMeander>();
    }
}
