//! Subordinator simulation

use crate::{SimulationError, XResult, random::stable, simulation::prelude::*};
use rayon::prelude::*;

/// alpha-stable subordinator
///
/// # Mathematical Formulation
///
/// A subordinator is a Lévy process that is non-negative and has a non-decreasing sample path.
#[derive(Debug, Clone)]
pub struct Subordinator {
    /// The stability index of the subordinator, whose value must be in the range (0, 1).
    alpha: f64,
}

impl Subordinator {
    /// Create a new `Subordinator`
    ///
    /// # Arguments
    ///
    /// * `alpha` - The stability index of the subordinator, whose value must be in the range (0, 1).
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::Subordinator;
    ///
    /// let subordinator = Subordinator::new(0.5).unwrap();
    /// ```
    pub fn new(alpha: f64) -> XResult<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `alpha` must be in the range (0, 1], got {alpha}"
            ))
            .into());
        }
        Ok(Self { alpha })
    }

    /// Get the stability index
    pub fn get_alpha(&self) -> f64 {
        self.alpha
    }
}

impl ContinuousProcess for Subordinator {
    /// Simulate subordinator
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the subordinator simulation.
    /// * `time_step` - The time step of the subordinator simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::Subordinator, prelude::*};
    ///
    /// let subordinator = Subordinator::new(0.5).unwrap();
    /// let (t, x) = subordinator.simulate(1.0, 0.1).unwrap();
    /// ```
    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_subordinator(self.alpha, duration, time_step)
    }
}

/// Simulate subordinator
///
/// # Arguments
///
/// * `alpha` - The stability index
/// * `duration` - The duration
/// * `time_step` - The time step
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::subordinator::simulate_subordinator;
///
/// let (t, x) = simulate_subordinator(0.5, 1.0, 0.1).unwrap();
/// ```
pub fn simulate_subordinator(
    alpha: f64,
    duration: f64,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    // if alpha <= 0.0 || alpha > 1.0 {
    //     return Err(SimulationError::InvalidParameters(format!(
    //         "The `alpha` must be in the range (0, 1], got {alpha}"
    //     ))
    //     .into());
    // }
    // if time_step <= 0.0 {
    //     return Err(SimulationError::InvalidParameters(format!(
    //         "The `time_step` must be positive, got {time_step}"
    //     ))
    //     .into());
    // }
    // if duration <= 0.0 {
    //     return Err(SimulationError::InvalidParameters(format!(
    //         "The `duration` must be positive, got `{duration}`"
    //     ))
    //     .into());
    // }
    // if time_step > duration {
    //     return Err(SimulationError::InvalidParameters(format!(
    //         "The `time_step` must be less than or equal to the `duration`, got `{time_step}` > `{duration}`"
    //     ))
    //     .into());
    // }

    let num_steps = (duration / time_step).ceil() as usize;
    let actual_time_step = duration / num_steps as f64;

    let mut t = Vec::with_capacity(num_steps + 1);

    let power = 1.0 / alpha;
    let dt_power = actual_time_step.powf(power);
    let noise = stable::skew_rands(alpha, num_steps)?
        .into_par_iter()
        .map(|x| x * dt_power)
        .collect::<Vec<_>>();

    let x = unsafe {
        let mut x = Vec::with_capacity(num_steps + 1);
        x.push(0.0);
        t.push(0.0);

        let mut sum = 0.0;
        for i in 0..num_steps {
            let current_t = (i + 1) as f64 * actual_time_step;
            sum += *noise.get_unchecked(i);
            x.push(sum);
            t.push(current_t);
        }

        x
    };

    if let Some(last_t) = t.last_mut() {
        *last_t = duration;
    }

    Ok((t, x))
}

/// Inverse alpha-stable subordinator
#[derive(Debug, Clone)]
pub struct InvSubordinator {
    /// The stability index
    alpha: f64,
}

impl InvSubordinator {
    /// Create a new `InvSubordinator`
    ///
    /// # Arguments
    ///
    /// * `alpha` - The stability index
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::InvSubordinator;
    ///
    /// let inv_subordinator = InvSubordinator::new(0.5).unwrap();
    /// ```
    pub fn new(alpha: f64) -> XResult<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `alpha` must be in the range (0, 1], got {alpha}"
            ))
            .into());
        }
        Ok(Self { alpha })
    }

    /// Get the stability index
    pub fn get_alpha(&self) -> f64 {
        self.alpha
    }
}

impl ContinuousProcess for InvSubordinator {
    /// Simulate inverse subordinator
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration
    /// * `time_step` - The time step
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::InvSubordinator, prelude::*};
    ///
    /// let inv_subordinator = InvSubordinator::new(0.5).unwrap();
    /// let (t, x) = inv_subordinator.simulate(1.0, 0.1).unwrap();
    /// ```
    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_invsubordinator(self.alpha, duration, time_step)
    }
}

/// Simulate inverse subordinator
///
/// # Arguments
///
/// * `alpha` - The stability index
/// * `duration` - The duration
/// * `time_step` - The time step
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::subordinator::simulate_invsubordinator;
///
/// let (t, x) = simulate_invsubordinator(0.5, 1.0, 0.1).unwrap();
/// ```
pub fn simulate_invsubordinator(
    alpha: f64,
    duration: f64,
    time_step: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    if alpha <= 0.0 || alpha > 1.0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `alpha` must be in the range (0, 1], got {alpha}"
        ))
        .into());
    }
    if time_step <= 0.0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `time_step` must be positive, got {time_step}"
        ))
        .into());
    }
    if duration <= 0.0 {
        return Err(SimulationError::InvalidParameters(format!(
            "The `duration` must be positive, got `{duration}`"
        ))
        .into());
    }
    if time_step > duration {
        return Err(SimulationError::InvalidParameters(format!(
            "The `time_step` must be less than or equal to the `duration`, got `{time_step}` > `{duration}`"
        ))
        .into());
    }
    let mut mut_duration = duration;
    let (t, s) = loop {
        let (t, s) = simulate_subordinator(alpha, mut_duration, time_step)?;
        let last = match s.last() {
            Some(x) => *x,
            None => return Err(SimulationError::Unknown.into()),
        };
        if last >= duration {
            break (t, s);
        }
        mut_duration *= 2.0;
    };

    // 计算逆过程的时间点，避免使用 linspace
    let num_inv_steps = (duration / time_step).ceil() as usize;
    let actual_inv_time_step = duration / num_inv_steps as f64;

    let mut inv_times = Vec::with_capacity(num_inv_steps + 1);
    let mut inv_path = Vec::with_capacity(num_inv_steps + 1);

    for i in 0..=num_inv_steps {
        inv_times.push(i as f64 * actual_inv_time_step);
    }

    // 确保最后一个时间点精确等于 duration
    if let Some(last_t) = inv_times.last_mut() {
        *last_t = duration;
    }

    for &target_time in &inv_times {
        let pos = match s.binary_search_by(|&x| x.partial_cmp(&target_time).unwrap()) {
            Ok(idx) => idx,
            Err(idx) => {
                if idx >= s.len() {
                    s.len() - 1
                } else {
                    idx
                }
            }
        };

        inv_path.push(t[pos]);
    }

    Ok((inv_times, inv_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_subordinator() {
        let subordinator = Subordinator::new(0.5).unwrap();
        let time_step = 0.1;
        let duration = 1.0;
        let (t, x) = subordinator.simulate(duration, time_step).unwrap();
        println!("t: {t:?}");
        println!("x: {x:?}");
    }

    #[test]
    fn test_fpt() {
        let subordinator = Subordinator::new(0.5).unwrap();
        let time_step = 0.1;
        let fpt = subordinator.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {fpt:?}");
    }

    #[test]
    fn test_occupation_time() {
        let subordinator = Subordinator::new(0.5).unwrap();
        let time_step = 0.1;
        let ot = subordinator
            .occupation_time((-1.0, 1.0), 10.0, time_step)
            .unwrap();
        println!("ot: {ot:?}");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Subordinator>();
    }

    #[test]
    fn test_inv_subordinator() {
        let alpha = 0.7;
        let duration = 5.0;
        let time_step = 0.1;

        let (inv_times, inv_path) = simulate_invsubordinator(alpha, duration, time_step).unwrap();
        println!("inv_times: {inv_times:?}");
        println!("inv_path: {inv_path:?}");

        // 验证单调性
        assert!(inv_path.windows(2).all(|w| w[0] <= w[1]));

        // 验证边界条件
        assert_eq!(inv_times[0], 0.0);
        assert_eq!(inv_path[0], 0.0);
        assert!(inv_times.last().unwrap() >= &duration);
    }
}
