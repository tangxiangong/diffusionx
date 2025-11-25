//! Subordinator simulation

use crate::{
    SimulationError, XResult, check_duration_time_step,
    random::stable::{self, sample_standard_alpha},
    simulation::prelude::*,
};
use rand::rng;
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
    fn start(&self) -> f64 {
        0.0
    }

    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_subordinator(self.alpha, duration, time_step)
    }

    fn displacement(&self, duration: f64, time_step: f64) -> XResult<f64> {
        check_duration_time_step(duration, time_step)?;

        let num_steps = (duration / time_step).ceil() as usize;
        let power = 1.0 / self.alpha;
        let sigma = time_step.powf(power);
        let generator = sample_standard_alpha;
        let mut delta_x = (0..num_steps - 1)
            .into_par_iter()
            .map_init(rng, |r, _| sigma * generator(self.alpha, 1.0, r))
            .sum();

        let last_step = duration - (num_steps - 1) as f64 * time_step;
        delta_x += generator(self.alpha, 1.0, &mut rng()) * last_step.powf(power);
        Ok(delta_x)
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
    check_duration_time_step(duration, time_step)?;

    let num_steps = (duration / time_step).ceil() as usize;
    let power = 1.0 / alpha;
    let sigma = time_step.powf(power);
    let noise = stable::skew_rands(alpha, num_steps - 1)?;

    let mut t = Vec::with_capacity(num_steps + 1);
    let mut x = Vec::with_capacity(num_steps + 1);

    t.push(0.0);
    x.push(0.0);

    let mut current_x = 0.0;
    let mut current_t = 0.0;
    for xi in noise {
        current_x += xi * sigma;
        x.push(current_x);
        current_t += time_step;
        t.push(current_t);
    }
    let last_step = duration - current_t;
    let xi = stable::skew_rand(alpha)?;
    let sigma = last_step.powf(power);
    current_x += xi * sigma;
    x.push(current_x);
    t.push(duration);

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
    fn start(&self) -> f64 {
        0.0
    }

    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_invsubordinator(self.alpha, duration, time_step)
    }

    fn displacement(&self, duration: f64, time_step: f64) -> XResult<f64> {
        let mut mut_duration = duration;
        let (t, s) = loop {
            let (t, s) = simulate_subordinator(self.alpha, mut_duration, time_step)?;
            let last = match s.last() {
                Some(x) => *x,
                None => return Err(SimulationError::Unknown.into()),
            };
            if last >= duration {
                break (t, s);
            }
            mut_duration *= 2.0;
        };

        let num_inv_steps = (duration / time_step).ceil() as usize;

        let mut inv_times = Vec::with_capacity(num_inv_steps + 1);

        for i in 0..=num_inv_steps - 1 {
            inv_times.push(i as f64 * time_step);
        }
        inv_times.push(duration);

        let target_time = duration;

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

        Ok(t[pos])
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
    check_duration_time_step(duration, time_step)?;

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

    let num_inv_steps = (duration / time_step).ceil() as usize;

    let mut inv_times = Vec::with_capacity(num_inv_steps + 1);
    let mut inv_path = Vec::with_capacity(num_inv_steps + 1);

    for i in 0..=num_inv_steps - 1 {
        inv_times.push(i as f64 * time_step);
    }
    inv_times.push(duration);

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
