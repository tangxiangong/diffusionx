use crate::{
    SimulationError, XResult, random::uniform::bool_rands, simulation::prelude::*, utils::cumsum,
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
///
/// # Fields
/// - `step_size`: The step size of the random walk.
/// - `probability`: The probability of the step in the positive direction.
/// - `start_position`: The starting position of the process.
#[derive(Clone, Debug)]
pub struct LatticeRandomWalk {
    step_size: f64,
    probability: f64,
    start_position: f64,
}

impl Default for LatticeRandomWalk {
    fn default() -> Self {
        Self::new(1.0, 0.5, 0.0).unwrap()
    }
}

impl LatticeRandomWalk {
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
                "probability must be between 0 and 1, got {}",
                probability
            ))
            .into());
        }
        if step_size <= 0.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "step size must be greater than 0, got {}",
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

    /// Get the step size of the lattice random walk
    ///
    /// # Returns
    ///
    /// A f64 representing the step size of the lattice random walk.
    ///
    /// # Example
    ///
    /// ```rust
    /// let rw = LatticeRandomWalk::new(0.5, 1.0, 0.0).unwrap();
    /// let step_size = rw.step_size();
    /// ```
    pub fn step_size(&self) -> f64 {
        self.step_size
    }

    /// Get the probability of the step in the positive direction
    ///
    /// # Returns
    ///
    /// A f64 representing the probability of the step in the positive direction.
    ///
    /// # Example
    ///
    /// ```rust
    /// let rw = LatticeRandomWalk::new(0.5, 1.0, 0.0).unwrap();
    /// let probability = rw.probability();
    /// ```
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Get the start position of the lattice random walk
    ///
    /// # Returns
    ///
    /// A f64 representing the start position of the lattice random walk.
    ///
    /// # Example
    ///
    /// ```rust
    /// let rw = LatticeRandomWalk::default();
    /// let start_position = rw.start_position();
    /// ```
    pub fn start_position(&self) -> f64 {
        self.start_position
    }

    /// Simulate the lattice random walk
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let rw = LatticeRandomWalk::default();
    /// let (t, x) = rw.simulate(1000).unwrap();
    /// ```
    pub fn simulate(&self, num_step: usize) -> XResult<DiscretePair> {
        simulate_lattice_random_walk(
            self.step_size,
            self.probability,
            self.start_position,
            num_step,
        )
    }

    /// Get the mean of the lattice random walk simulation
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    /// * `particles` - The number of particles.
    ///
    /// # Returns
    ///
    /// A f64 representing the mean of the lattice random walk simulation.  
    ///
    /// # Example
    ///
    /// ```rust
    /// let rw = LatticeRandomWalk::default();
    /// let mean = rw.mean(1000).unwrap();
    /// ```
    pub fn mean(&self, num_step: usize, particles: usize) -> XResult<f64> {
        let traj = self.step(num_step).unwrap();
        traj.raw_moment(1, particles, 0.1)
    }

    /// Get the mean square displacement of the lattice random walk simulation
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    /// * `particles` - The number of particles.
    ///
    /// # Returns
    ///
    /// A f64 representing the mean square displacement of the lattice random walk simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let rw = LatticeRandomWalk::default();
    /// let msd = rw.msd(1000).unwrap();
    /// ```
    pub fn msd(&self, num_step: usize, particles: usize) -> XResult<f64> {
        let traj = self.step(num_step).unwrap();
        traj.central_moment(2, particles, 0.1)
    }

    /// Get the raw moment of the lattice random walk
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    ///
    /// # Returns
    ///
    /// A f64 representing the raw moment of the lattice random walk simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let rw = LatticeRandomWalk::default();
    /// let moment = rw.raw_moment(1.0, 1000).unwrap();
    /// ```
    pub fn raw_moment(&self, num_step: usize, order: i32, particles: usize) -> XResult<f64> {
        let traj = self.step(num_step).unwrap();
        traj.raw_moment(order, particles, 0.1)
    }

    /// Get the central moment of the lattice random walk
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    ///
    /// # Returns
    ///
    /// A f64 representing the central moment of the lattice random walk.
    ///
    /// # Example
    ///
    /// ```rust
    /// let rw = LatticeRandomWalk::default();
    /// let msd = rw.msd(1.0, 1000).unwrap();
    /// ```
    pub fn central_moment(&self, num_step: usize, order: i32, particles: usize) -> XResult<f64> {
        let traj = self.step(num_step).unwrap();
        traj.central_moment(order, particles, 0.1)
    }
}

impl DiscreteProcess for LatticeRandomWalk {
    /// Simulate the lattice random walk
    ///
    /// # Arguments
    ///
    /// * `num_step` - The number of steps of the simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::LatticeRandomWalk;
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
    }
}
