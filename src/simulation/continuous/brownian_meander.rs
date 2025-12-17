//! Brownian meander simulation

use crate::{
    FloatExt, SimulationError, XResult, check_duration_time_step,
    random::PAR_THRESHOLD,
    simulation::{continuous::Bm, prelude::*},
    utils::float_eq,
};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Brownian meander
#[derive(Debug, Clone)]
pub struct BrownianMeander<T: FloatExt = f64> {
    _marker: std::marker::PhantomData<T>,
}

impl<T: FloatExt> Default for BrownianMeander<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatExt> BrownianMeander<T> {
    /// Create a new `BrownianMeander`
    ///
    ///  # Example
    ///
    ///  ```rust
    /// use diffusionx::simulation::continuous::BrownianMeander;
    ///
    /// let bm = BrownianMeander::new();
    /// ```
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }

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
    /// let bm = BrownianMeander::new();
    /// let fpt = bm.fpt((-1.0, 1.0), 0.1).unwrap();
    /// ```
    pub fn fpt(&self, domain: (T, T), time_step: T) -> XResult<Option<T>>
    where
        StandardNormal: Distribution<T>,
    {
        let (a, b) = domain;

        let (t, x) = self.simulate(T::one(), time_step)?;
        if let Some(index) = x.iter().position(|&x| x <= a || x >= b) {
            Ok(Some(t[index]))
        } else {
            Ok(None)
        }
    }
}

impl<T: FloatExt> ContinuousProcess<T> for BrownianMeander<T>
where
    StandardNormal: Distribution<T>,
{
    fn start(&self) -> T {
        T::zero()
    }

    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        simulate_brownian_meander(duration, time_step)
    }

    fn displacement(&self, duration: T, time_step: T) -> XResult<T> {
        if duration > T::one() {
            // Duration must be positive and not exceed 1.0
            return Err(SimulationError::InvalidParameters(format!(
                "The `duration` must be in (0.0, 1.0], got {duration:?}"
            ))
            .into());
        }

        let bm = Bm::default();
        let (bm_t, bm_traj) = bm.simulate(T::one(), time_step)?;

        let hint_indexes = bm_traj
            .iter()
            .enumerate()
            .filter(|(_, x)| float_eq(**x, T::zero()))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let last_hint_index = *hint_indexes.last().unwrap_or(&0);
        let tau = if last_hint_index == bm_t.len() - 1 {
            T::one() - time_step
        } else {
            bm_t[last_hint_index]
        };
        let coe = T::one() / (T::one() - tau).sqrt();
        if bm_t.len() < PAR_THRESHOLD {
            Ok(bm_t
                .iter()
                .map(|&t_i| {
                    let time = t_i * (T::one() - tau) + tau;
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
        } else {
            Ok(bm_t
                .par_iter()
                .map(|&t_i| {
                    let time = t_i * (T::one() - tau) + tau;
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
pub fn simulate_brownian_meander<T: FloatExt>(
    duration: T,
    time_step: T,
) -> XResult<(Vec<T>, Vec<T>)>
where
    StandardNormal: Distribution<T>,
{
    if duration > T::one() {
        // Duration must be positive and not exceed 1.0
        return Err(SimulationError::InvalidParameters(format!(
            "The `duration` must be in (0.0, 1.0], got {duration:?}"
        ))
        .into());
    }

    check_duration_time_step(duration, time_step)?;

    let bm = Bm::default();
    let (bm_t, bm_traj) = bm.simulate(T::one(), time_step)?;

    let hint_indexes = bm_traj
        .iter()
        .enumerate()
        .filter(|(_, x)| float_eq(**x, T::zero()))
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    let last_hint_index = *hint_indexes.last().unwrap_or(&0);
    let tau = if last_hint_index == bm_t.len() - 1 {
        T::one() - time_step
    } else {
        bm_t[last_hint_index]
    };
    let coe = T::one() / (T::one() - tau).sqrt();
    let x = bm_t
        .par_iter()
        .map(|&t_i| {
            let time = t_i * (T::one() - tau) + tau;
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
        let bm = BrownianMeander::new();
        let duration = 1.0;
        let time_step = 0.1;
        let (t, x) = bm.simulate(duration, time_step).unwrap();
        println!("t: {t:?}");
        println!("x: {x:?}");
    }

    #[test]
    fn test_raw_moment() {
        let bm = BrownianMeander::new();
        let duration = 1.0;
        let time_step = 0.1;
        let traj = bm.duration(duration).unwrap();
        let moment = traj.raw_moment(1, 1000, time_step).unwrap();
        println!("moment: {moment:?}");
    }

    #[test]
    fn test_fpt() {
        let bm = BrownianMeander::new();
        let time_step = 0.1;
        let fpt = bm.fpt((-1.0, 1.0), time_step).unwrap();
        println!("fpt: {fpt:?}");
    }

    #[test]
    fn test_occupation_time() {
        let bm = BrownianMeander::new();
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
