use crate::{
    SimulationError, XResult,
    random::{exponential, normal, stable},
    simulation::prelude::*,
    utils::cumsum,
};

/// Continuous time random walk
///
/// # Mathematical Formulation
/// A continuous time random walk (CTRW) is a stochastic process that generalizes random walks by introducing
/// random waiting times between jumps. Mathematically, it can be described as:
///
/// X(t) = \sum_{i=1}^{N(t)} J_i
///
/// where:
/// - X(t) is the position at time t
/// - J_i are random jump lengths (often from a symmetric distribution)
/// - N(t) is a counting process representing the number of jumps by time t
///
/// The waiting times between jumps typically follow a distribution with heavy tails, often
/// characterized by a power-law. When the waiting time distribution has infinite mean,
/// the resulting process exhibits subdiffusive behavior, with mean squared displacement
/// growing sublinearly with time: <X²(t)> ~ t^α where 0 < α < 1.
///
/// CTRWs are widely used to model anomalous diffusion in complex systems, including
/// transport in disordered media, financial time series, and biological processes.
///
/// # Fields
/// - `alpha`: Between 0 and 1, the exponent of the waiting time distribution, when `alpha = 1` the waiting time is exponential, otherwise it is a power-law with tail index `alpha`.
/// - `beta`: Between 0 and 2, the exponent of the jump length distribution, when `beta = 2` the jump length is normal, otherwise it is a power-law with tail index `beta`.
/// - `start_position`: The starting position of the process.
#[derive(Clone, Debug)]
pub struct CTRW {
    alpha: f64,
    beta: f64,
    start_position: f64,
}

impl Default for CTRW {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 2.0,
            start_position: 0.0,
        }
    }
}

impl CTRW {
    pub fn new(
        alpha: impl Into<f64>,
        beta: impl Into<f64>,
        start_position: impl Into<f64>,
    ) -> XResult<Self> {
        let alpha = alpha.into();
        let beta = beta.into();
        let start_position = start_position.into();
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "alpha must be between 0 and 1, got {}",
                alpha
            ))
            .into());
        }
        if beta <= 0.0 || beta > 2.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "beta must be between 0 and 2, got {}",
                beta
            ))
            .into());
        }
        Ok(Self {
            alpha,
            beta,
            start_position,
        })
    }

    /// Get the alpha of the continuous time random walk
    ///
    /// # Returns
    ///
    /// A f64 representing the alpha of the continuous time random walk.
    ///
    /// # Example
    ///
    /// ```rust
    /// let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
    /// let alpha = ctrw.alpha();
    /// ```
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the beta of the continuous time random walk
    ///
    /// # Returns
    ///
    /// A f64 representing the beta of the continuous time random walk.
    ///
    /// # Example
    ///
    /// ```rust
    /// let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
    /// let beta = ctrw.beta();
    /// ```
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Get the start position of the continuous time random walk
    ///
    /// # Returns
    ///
    /// A f64 representing the start position of the continuous time random walk.
    ///
    /// # Example
    ///
    /// ```rust
    /// let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
    /// let start_position = ctrw.start_position();
    /// ```
    pub fn start_position(&self) -> f64 {
        self.start_position
    }

    /// Simulate the continuous time random walk
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
    /// let (t, x) = ctrw.simulate(10.0).unwrap();
    /// ```
    pub fn simulate(&self, duration: impl Into<f64>) -> XResult<Pair> {
        simulate_ctrw_with_duration(self.alpha, self.beta, duration, self.start_position)
    }

    /// Simulate the continuous time random walk
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
    /// let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
    /// let (t, x) = ctrw.simulate_with_step(1000).unwrap();
    /// ```
    pub fn simulate_with_step(&self, num_step: usize) -> XResult<Pair> {
        simulate_ctrw_with_step(self.alpha, self.beta, num_step, self.start_position)
    }

    /// Get the mean of the continuous time random walk simulation
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `particles` - The number of particles.
    ///
    /// # Returns
    ///
    /// A f64 representing the mean of the continuous time random walk simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
    /// let mean = ctrw.mean(1.0, 1000).unwrap();
    /// ```
    pub fn mean(&self, duration: impl Into<f64>, particles: usize) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.raw_moment(1, particles, 0.1)
    }

    /// Get the mean square displacement of the continuous time random walk simulation
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `particles` - The number of particles.
    ///
    /// # Returns
    ///
    /// A f64 representing the mean square displacement of the continuous time random walk simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
    /// let msd = ctrw.msd(1.0, 1000).unwrap();
    /// ```
    pub fn msd(&self, duration: impl Into<f64>, particles: usize) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.central_moment(2, particles, 0.1)
    }

    /// Get the raw moment of the continuous time random walk simulation
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the continuous time random walk simulation.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    ///
    /// # Returns
    ///
    /// A f64 representing the raw moment of the continuous time random walk simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
    /// let moment = ctrw.raw_moment(1.0, 1000).unwrap();
    /// ```
    pub fn raw_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
    ) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.raw_moment(order, particles, 0.1)
    }

    /// Get the central moment of the continuous time random walk simulation
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the continuous time random walk simulation.
    /// * `order` - The order of the moment.
    /// * `particles` - The number of particles.
    ///
    /// # Returns
    ///
    /// A f64 representing the central moment of the continuous time random walk simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
    /// let msd = ctrw.msd(1.0, 1000).unwrap();
    /// ```
    pub fn central_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
    ) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.central_moment(order, particles, 0.1)
    }

    /// Get the first passage time of the continuous time random walk simulation
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the continuous time random walk simulation.
    /// * `max_duration` - The maximum duration of the continuous time random walk simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the first passage time of the continuous time random walk simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
    /// let fpt = ctrw.fpt((-1.0, 1.0), 1000.0).unwrap();
    /// ```
    pub fn fpt(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        max_duration: impl Into<f64>,
    ) -> XResult<Option<f64>> {
        let fpt = FirstPassageTime::new(self, domain)?;
        fpt.simulate(max_duration, 0.1)
    }

    /// Get the occupation time of the continuous time random walk simulation
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain of the continuous time random walk simulation.
    /// * `duration` - The duration of the continuous time random walk simulation.
    ///
    /// # Returns
    ///
    /// A f64 representing the occupation time of the continuous time random walk simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
    /// let ot = ctrw.occupation_time((-1.0, 1.0), 1000.0).unwrap();
    /// ```
    pub fn occupation_time(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        duration: impl Into<f64>,
    ) -> XResult<f64> {
        let ot = OccupationTime::new(self, domain, duration)?;
        ot.simulate(0.1)
    }
}

impl ContinuousProcess for CTRW {
    /// Simulate the continuous time random walk
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the simulation.
    /// * `time_step` - The time step of the simulation.
    ///
    /// # Returns
    ///
    /// A tuple containing the time and the position of the simulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::CTRW;
    /// let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
    /// let (t, x) = ctrw.simulate(10.0, 0.1).unwrap();
    /// ```
    fn simulate(&self, duration: impl Into<f64>, _time_step: f64) -> XResult<Pair> {
        simulate_ctrw_with_duration(self.alpha, self.beta, duration, self.start_position)
    }
}

pub fn simulate_ctrw_with_step(
    alpha: f64,
    beta: f64,
    num_step: usize,
    start_position: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let waiting_times = if alpha == 1.0 {
        exponential::rands(1.0, num_step)?
    } else {
        stable::skew_rands(alpha, num_step)?
    };
    let jump_lengths = if beta == 2.0 {
        normal::standard_rands(num_step)
    } else {
        stable::sym_standard_rands(beta, num_step)?
    };
    let t = cumsum(0.0, &waiting_times);
    let x = cumsum(start_position, &jump_lengths);
    Ok((t, x))
}

pub fn simulate_ctrw_with_duration(
    alpha: f64,
    beta: f64,
    duration: impl Into<f64>,
    start_position: f64,
) -> XResult<(Vec<f64>, Vec<f64>)> {
    let duration = duration.into();
    let mut num_step = duration.ceil() as usize;
    let (t, x) = loop {
        let (t, x) = simulate_ctrw_with_step(alpha, beta, num_step, start_position)?;
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
        x_[index] = x_[index - 1];
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
    fn test_simulate_ctrw_with_step() {
        let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
        let (t, x) = ctrw.simulate_with_step(1000).unwrap();
        assert_eq!(t.len(), 1001);
        assert_eq!(x.len(), 1001);
    }

    #[test]
    fn test_simulate_ctrw_with_duration() {
        let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
        let (_t, _x) = ctrw.simulate(10.0).unwrap();
    }

    #[test]
    fn test_mean() {
        let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
        let _mean = ctrw.mean(1.0, 1000).unwrap();
    }

    #[test]
    fn test_msd() {
        let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
        let _msd = ctrw.msd(1.0, 1000).unwrap();
    }

    #[test]
    fn test_raw_moment() {
        let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
        let _moment = ctrw.raw_moment(1.0, 1, 1000).unwrap();
    }

    #[test]
    fn test_central_moment() {
        let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
        let _moment = ctrw.central_moment(1.0, 2, 1000).unwrap();
    }

    #[test]
    fn test_fpt() {
        let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
        let _fpt = ctrw.fpt((-1.0, 1.0), 1000.0).unwrap();
    }

    #[test]
    fn test_occupation_time() {
        let ctrw = CTRW::new(0.5, 1.0, 0.0).unwrap();
        let ot = ctrw.occupation_time((-1.0, 1.0), 1000.0).unwrap();
        assert!((0.0..=1000.0).contains(&ot));
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CTRW>();
    }
}
