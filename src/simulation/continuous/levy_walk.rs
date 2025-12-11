use crate::{
    FloatExt, SimulationError, XResult, check_duration_time_step,
    random::{exponential, stable},
    simulation::prelude::*,
    utils::{cumsum, linear_interpolate},
};
use num_traits::FloatConst;
use rand::{prelude::*, rng};
use rand_distr::{Exp1, uniform::SampleUniform};
use rayon::prelude::*;

/// Lévy walk
///
/// # Mathematical Formulation
///
/// A Lévy walk is a random walk model where the walker moves with a constant velocity
/// between turning points. At each turning point, the walker randomly chooses a new
/// direction and a new flight time $\tau$ from a probability distribution $\psi(\tau) ∼ \tau^{-1-\alpha}$
/// with 0 < $\alpha$ < 1. The flight length is proportional to the flight time: $l = v\tau$,
/// where $v$ is the constant velocity.
#[derive(Clone, Debug)]
pub struct LevyWalk<T: FloatExt = f64> {
    /// The waiting time distribution exponent
    alpha: T,
    /// The velocity
    velocity: T,
    /// The starting position
    start_position: T,
}

impl<T: FloatExt> Default for LevyWalk<T> {
    fn default() -> Self {
        Self {
            alpha: T::one(),
            velocity: T::one(),
            start_position: T::zero(),
        }
    }
}

impl<T: FloatExt> LevyWalk<T> {
    /// Create a new `LevyWalk`
    ///
    /// # Arguments
    ///
    /// * `alpha` - The alpha of the Levy walk.
    /// * `velocity` - The velocity of the Levy walk.
    /// * `start_position` - The starting position of the Levy walk.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::LevyWalk;
    ///
    /// let levy_walk = LevyWalk::new(0.5, 1.0, 0.0).unwrap();
    /// ```
    pub fn new(alpha: T, velocity: T, start_position: T) -> XResult<Self> {
        if alpha <= T::zero() || alpha > T::one() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `alpha` must be between 0 and 1, got {alpha:?}"
            ))
            .into());
        }
        if velocity <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `velocity` must be positive, got {velocity:?}"
            ))
            .into());
        }
        Ok(Self {
            alpha,
            velocity,
            start_position,
        })
    }

    /// Get the waiting time distribution exponent
    pub fn get_alpha(&self) -> T {
        self.alpha
    }

    /// Get the velocity
    pub fn get_velocity(&self) -> T {
        self.velocity
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> T {
        self.start_position
    }

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
    /// use diffusionx::simulation::{continuous::LevyWalk, prelude::*};
    ///
    /// let levy_walk = LevyWalk::new(0.5, 1.0, 0.0).unwrap();
    /// let (t, x) = levy_walk.simulate(10.0, 0.1).unwrap();
    /// ```
    pub fn simulate_with_step(&self, num_step: usize) -> XResult<(Vec<T>, Vec<T>)>
    where
        T: FloatConst + SampleUniform,
        Exp1: Distribution<T>,
    {
        simulate_levy_walk_with_step(self.alpha, self.velocity, num_step, self.start_position)
    }

    /// Simulate the Lévy walk with duration
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    pub fn simulate_with_duration(&self, duration: T) -> XResult<(Vec<T>, Vec<T>)>
    where
        T: FloatConst + SampleUniform,
        Exp1: Distribution<T>,
    {
        simulate_levy_walk_with_duration(self.alpha, self.velocity, duration, self.start_position)
    }
}

impl<T: FloatExt + FloatConst + SampleUniform> ContinuousProcess<T> for LevyWalk<T>
where
    Exp1: Distribution<T>,
{
    fn start(&self) -> T {
        self.start_position
    }

    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        let (t, x) = self.simulate_with_duration(duration)?;
        linear_interpolate(&t, &x, time_step)
    }

    fn displacement(&self, duration: T, _time_step: T) -> XResult<T> {
        check_duration_time_step(duration, _time_step)?;

        let mut num_step = duration.ceil().to_usize().unwrap();
        let (t, x) = loop {
            let (t, x) = simulate_levy_walk_with_step(
                self.alpha,
                self.velocity,
                num_step,
                self.start_position,
            )?;
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
        let index_x = unsafe { *x.get_unchecked(index) };
        let index_t = unsafe { *t.get_unchecked(index) };
        let index_x_pre = unsafe { *x.get_unchecked(index - 1) };
        let index_t_pre = unsafe { *t.get_unchecked(index - 1) };
        let last_x = if index_t > duration {
            let direction = if index_x >= index_x_pre {
                self.velocity
            } else {
                -self.velocity
            };
            index_x_pre + (duration - index_t_pre) * direction
        } else {
            index_x
        };
        Ok(last_x - self.start_position)
    }
}

/// Simulate the Lévy walk with step
///
/// # Arguments
///
/// * `alpha` - The waiting time distribution exponent.
/// * `velocity` - The velocity.
/// * `num_step` - The number of steps.
/// * `start_position` - The starting position.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::levy_walk::simulate_levy_walk_with_step;
///
/// let (t, x) = simulate_levy_walk_with_step(0.5, 1.0, 1000, 0.0).unwrap();
/// ```
pub fn simulate_levy_walk_with_step<T: FloatExt + FloatConst + SampleUniform>(
    alpha: T,
    velocity: T,
    num_step: usize,
    start_position: T,
) -> XResult<(Vec<T>, Vec<T>)>
where
    Exp1: Distribution<T>,
{
    let waiting_times = if alpha == T::one() {
        exponential::standard_rands(num_step)
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
        .map(|(&waiting_time, direction)| waiting_time * direction)
        .collect::<Vec<_>>();
    let t = cumsum(T::zero(), &waiting_times);
    let x = cumsum(start_position, &jump_lengths);
    Ok((t, x))
}

/// Simulate the Lévy walk with duration
///
/// # Arguments
///
/// * `alpha` - The waiting time distribution exponent.
/// * `velocity` - The velocity.
/// * `duration` - The duration.
/// * `start_position` - The starting position.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::levy_walk::simulate_levy_walk_with_duration;
///
/// let (t, x) = simulate_levy_walk_with_duration(0.5, 1.0, 10.0, 0.0).unwrap();
/// ```
pub fn simulate_levy_walk_with_duration<T: FloatExt + FloatConst + SampleUniform>(
    alpha: T,
    velocity: T,
    duration: T,
    start_position: T,
) -> XResult<(Vec<T>, Vec<T>)>
where
    Exp1: Distribution<T>,
{
    check_duration_time_step(duration, duration / T::from(2).unwrap())?;

    let mut num_step = duration.ceil().to_usize().unwrap();
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
    let mut t_ = vec![T::zero(); index + 1];
    let mut x_ = vec![T::zero(); index + 1];
    t_[..index].copy_from_slice(&t[..index]);
    x_[..index].copy_from_slice(&x[..index]);
    if t[index] > duration {
        t_[index] = duration;
        let direction = if x[index] >= x[index - 1] {
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
        let (_t, _x) = levy_walk.simulate_with_duration(10.0).unwrap();
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
