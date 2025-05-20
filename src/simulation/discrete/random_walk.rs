//! Random walk simulation

use crate::{
    SimulationError, XResult,
    random::{exponential, stable, uniform::bool_rands},
    simulation::prelude::*,
    utils::cumsum,
};
use rayon::prelude::*;

/// Lattice random walk
///
/// # Mathematical Formulation
///
/// A lattice random walk is a stochastic process that describes a path consisting of a succession of constant steps.
/// Mathematically, it can be represented as:
///
/// X_n = X_0 + d_n^{(p)} \sum_{i=1}^{n} a
///
/// where:
/// - X_n is the position after n steps
/// - X_0 is the initial position
/// - a is the step size
/// - d_n^{(p)} is the direction of the step, which is either +1 or -1 with probability p or 1-p respectively
#[derive(Clone, Debug)]
pub struct LatticeRandomWalk {
    /// The step size
    step_size: f64,
    /// The probability of the step in the positive direction
    probability: f64,
    /// The starting position
    start_position: f64,
}

impl Default for LatticeRandomWalk {
    fn default() -> Self {
        Self {
            step_size: 1.0,
            probability: 0.5,
            start_position: 0.0,
        }
    }
}

impl LatticeRandomWalk {
    /// Create a new `LatticeRandomWalk`
    ///
    /// # Arguments
    ///
    /// * `step_size` - The step size
    /// * `probability` - The probability of the step in the positive direction
    /// * `start_position` - The starting position
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::LatticeRandomWalk;
    ///
    /// let rw = LatticeRandomWalk::new(1.0, 0.5, 0.0).unwrap();
    /// ```
    pub fn new(
        step_size: impl Into<f64>,
        probability: impl Into<f64>,
        start_position: impl Into<f64>,
    ) -> XResult<Self> {
        let step_size = step_size.into();
        let probability = probability.into();
        let start_position = start_position.into();
        if probability <= 0.0 || probability > 1.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `probability` must be between 0 and 1, got {}",
                probability
            ))
            .into());
        }
        if step_size <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `step_size` must be greater than 0, got {}",
                step_size
            ))
            .into());
        }
        Ok(Self {
            step_size,
            probability,
            start_position,
        })
    }

    /// Get the step size
    pub fn step_size(&self) -> f64 {
        self.step_size
    }

    /// Get the probability of the step in the positive direction
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Get the starting position
    pub fn start_position(&self) -> f64 {
        self.start_position
    }
}

impl DiscreteProcess for LatticeRandomWalk {
    /// Simulate the lattice random walk
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{discrete::LatticeRandomWalk, prelude::*};
    ///
    /// let rw = LatticeRandomWalk::default();
    /// let (t, x) = rw.simulate(1000).unwrap();
    /// ```
    fn simulate(&self, num_step: usize) -> XResult<DiscretePair> {
        simulate_lattice_random_walk(
            self.step_size,
            self.probability,
            self.start_position,
            num_step,
        )
    }
}

/// Simulate the lattice random walk
///
/// # Arguments
///
/// * `step_size` - The step size
/// * `probability` - The probability of the step in the positive direction
/// * `start_position` - The starting position
/// * `num_step` - The number of steps
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::lattice_random_walk::simulate_lattice_random_walk;
///
/// let (t, x) = simulate_lattice_random_walk(0.5, 0.5, 0.0, 1000).unwrap();
/// ```
pub fn simulate_lattice_random_walk(
    step_size: f64,
    probability: f64,
    start_position: f64,
    num_step: usize,
) -> XResult<(Vec<usize>, Vec<f64>)> {
    let delta_x: Vec<f64> = bool_rands(probability, num_step)?
        .into_par_iter()
        .map(|x| if x { step_size } else { -step_size })
        .collect();
    let t = (0..=num_step).collect();
    let x = cumsum(start_position, &delta_x);
    Ok((t, x))
}

/// Random walk
///
/// # Mathematical Formulation
///
/// A stable random walk is a stochastic process that describes a path consisting of a succession of steps.
/// Mathematically, it can be represented as:
///
/// X_n = X_0 + \sum_{i=1}^{n} a_i
///
/// where:
/// - X_n is the position after n steps
/// - X_0 is the initial position
/// - a_i is the step size
#[derive(Clone, Debug)]
pub struct RandomWalk {
    /// The probability of the step in the positive direction
    probability: f64,
    /// The alpha parameter of the stable distribution
    alpha: f64,
    /// The starting position
    start_position: f64,
}

impl Default for RandomWalk {
    fn default() -> Self {
        Self {
            probability: 0.5,
            alpha: 2.0,
            start_position: 0.0,
        }
    }
}

impl RandomWalk {
    /// Create a new `RandomWalk`
    ///
    /// # Arguments
    ///
    /// * `probability` - The probability of the step in the positive direction
    /// * `alpha` - The alpha parameter of the stable distribution
    /// * `start_position` - The starting position
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::RandomWalk;
    ///
    /// let rw = RandomWalk::new(0.5, 1.0, 0.0).unwrap();
    /// ```
    pub fn new(
        probability: impl Into<f64>,
        alpha: impl Into<f64>,
        start_position: impl Into<f64>,
    ) -> XResult<Self> {
        let probability = probability.into();
        let alpha = alpha.into();
        let start_position = start_position.into();
        if probability <= 0.0 || probability > 1.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `probability` must be between 0 and 1, got {}",
                probability
            ))
            .into());
        }
        if alpha <= 0.0 || alpha > 2.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `alpha` must be between 0 and 2, got {}",
                alpha
            ))
            .into());
        }
        Ok(Self {
            probability,
            alpha,
            start_position,
        })
    }

    /// Get the probability of the step in the positive direction
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Get the alpha parameter of the stable distribution
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the starting position
    pub fn start_position(&self) -> f64 {
        self.start_position
    }
}

impl DiscreteProcess for RandomWalk {
    /// Simulate the random walk
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{discrete::RandomWalk, prelude::*};
    ///
    /// let rw = RandomWalk::default();
    /// let (t, x) = rw.simulate(1000).unwrap();
    /// ```
    fn simulate(&self, num_step: usize) -> XResult<DiscretePair> {
        simulate_random_walk(self.probability, self.alpha, self.start_position, num_step)
    }
}

/// Simulate the random walk
///
/// # Arguments
///
/// * `probability` - The probability of the step in the positive direction
/// * `alpha` - The alpha parameter of the stable distribution
/// * `start_position` - The starting position
/// * `num_step` - The number of steps
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::random_walk::simulate_random_walk;
///
/// let (t, x) = simulate_random_walk(0.5, 1.0, 0.0, 1000).unwrap();
/// ```
pub fn simulate_random_walk(
    probability: f64,
    alpha: f64,
    start_position: f64,
    num_step: usize,
) -> XResult<(Vec<usize>, Vec<f64>)> {
    let jump_lengths = if alpha == 1.0 {
        exponential::rands(1.0, num_step)?
    } else {
        stable::skew_rands(alpha, num_step)?
    };
    let delta_x: Vec<f64> = bool_rands(probability, num_step)?
        .into_par_iter()
        .zip(jump_lengths)
        .map(|(x, jump_length)| {
            if x {
                jump_length.abs()
            } else {
                -jump_length.abs()
            }
        })
        .collect();
    let t = (0..=num_step).collect();
    let x = cumsum(start_position, &delta_x);
    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_lattice_random_walk() {
        let rw = LatticeRandomWalk::default();
        let (t, x) = rw.simulate(1000).unwrap();
        assert_eq!(t.len(), 1001);
        assert_eq!(x.len(), 1001);
    }

    #[test]
    fn test_mean() {
        let rw = LatticeRandomWalk::default();
        let _mean = rw.mean(1000, 1000).unwrap();
    }

    #[test]
    fn test_msd() {
        let rw = LatticeRandomWalk::default();
        let _msd = rw.msd(1000, 1000).unwrap();
    }

    #[test]
    fn test_raw_moment() {
        let rw = LatticeRandomWalk::default();
        let _moment = rw.raw_moment(1000, 1, 1000).unwrap();
    }

    #[test]
    fn test_central_moment() {
        let rw = LatticeRandomWalk::default();
        let _moment = rw.central_moment(1000, 2, 1000).unwrap();
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LatticeRandomWalk>();
        assert_send_sync::<RandomWalk>();
    }
}
