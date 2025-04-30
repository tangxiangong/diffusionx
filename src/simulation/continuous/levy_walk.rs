use crate::{
    SimulationError, XResult,
    random::{exponential, stable},
    simulation::prelude::*,
    utils::cumsum,
};
use rand::{prelude::*, rng};
use rayon::prelude::*;

/// Lévy walk
///
/// # Mathematical Formulation
///
/// A Lévy walk is a random walk model where the walker moves with a constant velocity
/// between turning points. At each turning point, the walker randomly chooses a new
/// direction and a new flight time τ from a probability distribution ψ(τ) ~ τ^(-1-α)
/// with 0 < α < 1. The flight length is proportional to the flight time: l = vτ,
/// where v is the constant velocity.
///
/// # Fields
/// - `alpha`: Between 0 and 1, the exponent of the waiting time distribution, when `alpha = 1` the waiting time is exponential, otherwise it is a power-law with tail index `alpha`.
/// - `velocity`: The velocity of the walker.
/// - `start_position`: The starting position of the process.
#[derive(Clone, Debug)]
pub struct LevyWalk {
    alpha: f64,
    velocity: f64,
    start_position: f64,
}

impl Default for LevyWalk {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            velocity: 1.0,
            start_position: 0.0,
        }
    }
}

impl LevyWalk {
    /// Create a new LevyWalk
    ///
    /// # Arguments
    ///
    /// - `alpha`: The alpha of the Levy walk.
    /// - `velocity`: The velocity of the Levy walk.
    /// - `start_position`: The starting position of the Levy walk.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::LevyWalk;
    /// let levy_walk = LevyWalk::new(0.5, 1.0, 0.0).unwrap();
    /// ```
    pub fn new(
        alpha: impl Into<f64>,
        velocity: impl Into<f64>,
        start_position: impl Into<f64>,
    ) -> XResult<Self> {
        let alpha = alpha.into();
        let velocity = velocity.into();
        let start_position = start_position.into();
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "alpha must be between 0 and 1, got {}",
                alpha
            ))
            .into());
        }
        if velocity <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "velocity must be positive, got {}",
                velocity
            ))
            .into());
        }
        Ok(Self {
            alpha,
            velocity,
            start_position,
        })
    }

    /// Get the alpha of the Levy walk
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the velocity of the Levy walk
    pub fn velocity(&self) -> f64 {
        self.velocity
    }

    /// Get the start position of the Lévy walk
    pub fn start_position(&self) -> f64 {
        self.start_position
    }

    /// Simulate the Lévy walk with step
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::LevyWalk;
    /// let levy_walk = LevyWalk::new(0.5, 1.0, 0.0).unwrap();
    /// let (t, x) = levy_walk.simulate_with_step(1000).unwrap();
    /// ```
    pub fn simulate_with_step(&self, num_step: usize) -> XResult<Pair> {
        simulate_levy_walk_with_step(self.alpha, self.velocity, num_step, self.start_position)
    }
}

/// impl `ContinuousProcess` trait for LevyWalk
impl ContinuousProcess for LevyWalk {
    /// Simulate the Lévy walk
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::LevyWalk;
    /// let levy_walk = LevyWalk::new(0.5, 1.0, 0.0).unwrap();
    /// let (t, x) = levy_walk.simulate(10.0, 0.1).unwrap();
    /// ```
    fn simulate(&self, duration: impl Into<f64>, _time_step: f64) -> XResult<Pair> {
        simulate_levy_walk_with_duration(self.alpha, self.velocity, duration, self.start_position)
    }
}

/// Simulate the Lévy walk with step
///
/// # Arguments
///
/// * `alpha` - The alpha of the Lévy walk.
/// * `velocity` - The velocity of the Lévy walk.
/// * `num_step` - The number of steps of the Lévy walk.
/// * `start_position` - The starting position of the Lévy walk.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::levy_walk::simulate_levy_walk_with_step;
/// let (t, x) = simulate_levy_walk_with_step(0.5, 1.0, 1000, 0.0).unwrap();
/// ```
pub fn simulate_levy_walk_with_step(
    alpha: f64,
    velocity: f64,
    num_step: usize,
    start_position: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let waiting_times = if alpha == 1.0 {
        exponential::rands(1.0, num_step)?
    } else {
        stable::skew_rands(alpha, num_step)?
    };
    let directions = (0..num_step)
        .into_par_iter()
        .map_init(rng, |r, _| {
            if r.random_bool(0.5) {
                velocity
            } else {
                -velocity
            }
        })
        .collect::<Vec<_>>();
    let jump_lengths = waiting_times
        .par_iter()
        .zip(directions)
        .map(|(waiting_time, direction)| waiting_time * direction)
        .collect::<Vec<_>>();
    let t = cumsum(0.0, &waiting_times);
    let x = cumsum(start_position, &jump_lengths);
    Ok((t, x))
}

/// Simulate the Lévy walk with duration
///
/// # Arguments
///
/// * `alpha` - The alpha of the Lévy walk.
/// * `velocity` - The velocity of the Lévy walk.
/// * `duration` - The duration of the Lévy walk.
/// * `start_position` - The starting position of the Lévy walk.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::levy_walk::simulate_levy_walk_with_duration;
/// let (t, x) = simulate_levy_walk_with_duration(0.5, 1.0, 10.0, 0.0).unwrap();
/// ```
pub fn simulate_levy_walk_with_duration(
    alpha: f64,
    velocity: f64,
    duration: impl Into<f64>,
    start_position: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let duration = duration.into();
    let mut num_step = duration.ceil() as usize;
    let (t, x) = loop {
        let (t, x) = simulate_levy_walk_with_step(alpha, velocity, num_step, start_position)?;
        if t.last().is_none() {
            return Err(SimulationError::Unknown.into());
        }
        let end_time = *t.last().unwrap();
        if end_time >= duration {
            break (t, x);
        }
        num_step *= 2;
    };
    let index = t.iter().position(|&time| time >= duration).unwrap();
    let mut t_ = vec![0.0; index + 1];
    let mut x_ = vec![0.0; index + 1];
    t_[..index].copy_from_slice(&t[..index]);
    x_[..index].copy_from_slice(&x[..index]);
    if t[index] > duration {
        t_[index] = duration;
        let direction = if rng().random_bool(0.5) {
            velocity
        } else {
            -velocity
        };
        x_[index] = x[index - 1] + (duration - t[index - 1]) * direction;
    } else {
        t_[index] = t[index];
        x_[index] = x[index];
    }
    Ok((t_, x_))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_levy_walk_with_step() {
        let levy_walk = LevyWalk::new(0.5, 1.0, 0.0).unwrap();
        let (t, x) = levy_walk.simulate_with_step(1000).unwrap();
        assert_eq!(t.len(), 1001);
        assert_eq!(x.len(), 1001);
    }

    #[test]
    fn test_simulate_levy_walk_with_duration() {
        let levy_walk = LevyWalk::new(0.5, 1.0, 0.0).unwrap();
        let (_t, _x) = levy_walk.simulate(10.0, 0.1).unwrap();
    }

    #[test]
    fn test_mean() {
        let levy_walk = LevyWalk::new(0.5, 1.0, 0.0).unwrap();
        let _mean = levy_walk.mean(1.0, 1000, 0.1).unwrap();
    }

    #[test]
    fn test_msd() {
        let levy_walk = LevyWalk::new(0.5, 1.0, 0.0).unwrap();
        let _msd = levy_walk.msd(1.0, 1000, 0.1).unwrap();
    }

    #[test]
    fn test_raw_moment() {
        let levy_walk = LevyWalk::new(0.5, 1.0, 0.0).unwrap();
        let _moment = levy_walk.raw_moment(1.0, 1, 1000, 0.1).unwrap();
    }

    #[test]
    fn test_central_moment() {
        let levy_walk = LevyWalk::new(0.5, 1.0, 0.0).unwrap();
        let _moment = levy_walk.central_moment(1.0, 2, 1000, 0.1).unwrap();
    }

    #[test]
    fn test_fpt() {
        let levy_walk = LevyWalk::new(0.5, 1.0, 0.0).unwrap();
        let _fpt = levy_walk.fpt((-1.0, 1.0), 1000.0, 0.1).unwrap();
    }

    #[test]
    fn test_occupation_time() {
        let levy_walk = LevyWalk::new(0.5, 1.0, 0.0).unwrap();
        let ot = levy_walk.occupation_time((-1.0, 1.0), 1000.0, 0.1).unwrap();
        assert!((0.0..=1000.0).contains(&ot));
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LevyWalk>();
    }
}
