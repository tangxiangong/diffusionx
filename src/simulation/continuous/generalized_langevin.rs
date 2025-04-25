//! Generalized Langevin equation and subordinated Langevin equation simulation
//!

use crate::{
    SimulationError, XResult,
    random::{normal, stable},
    simulation::{continuous::Subordinator, prelude::*},
};
use rayon::prelude::*;

/// Generalized Langevin equation
///
/// dx(t) = f(x(t), t) dt + g(x(t), t) dL_alpha(t), x(0) = x0
///
/// where L_alpha(t) is the alpha-stable process.
///
/// # Fields
///
/// - `drift_func`: the drift function of the Generalized Langevin equation, f(x, t).
/// - `diffusion_func`: the diffusion function of the Generalized Langevin equation, g(x, t).
/// - `start_position`: the starting position of the Generalized Langevin equation, x0.
/// - `alpha`: the stability index of the alpha-stable process.
#[derive(Clone)]
pub struct GeneralizedLangevin<D, G>
where
    D: Fn(f64, f64) -> f64 + Clone + Send + Sync,
    G: Fn(f64, f64) -> f64 + Clone + Send + Sync,
{
    drift_func: D,
    diffusion_func: G,
    start_position: f64,
    alpha: f64,
}

impl<D, G> GeneralizedLangevin<D, G>
where
    D: Fn(f64, f64) -> f64 + Clone + Send + Sync,
    G: Fn(f64, f64) -> f64 + Clone + Send + Sync,
{
    pub fn new(
        drift_func: D,
        diffusion_func: G,
        start_position: impl Into<f64>,
        alpha: impl Into<f64>,
    ) -> XResult<Self> {
        let start_position = start_position.into();
        let alpha = alpha.into();
        if alpha <= 0.0 || alpha > 2.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `alpha` must be in the range (0, 2], got {}",
                alpha
            ))
            .into());
        }
        Ok(Self {
            drift_func,
            diffusion_func,
            start_position,
            alpha,
        })
    }

    /// Simulate the Generalized Langevin equation
    ///
    /// # Arguments
    ///
    /// - `duration`: the duration of the simulation
    /// - `time_step`: the time step of the simulation
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::GeneralizedLangevin, prelude::*};
    /// let langevin = GeneralizedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 1.7)
    ///     .unwrap();
    /// let (t, x) = langevin.simulate(1.0, 0.01).unwrap();
    /// ```
    pub fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        simulate_generalized_langevin(
            &self.drift_func,
            &self.diffusion_func,
            self.start_position(),
            self.alpha,
            duration,
            time_step,
        )
    }

    /// Get the starting position of the Generalized Langevin equation
    pub fn start_position(&self) -> f64 {
        self.start_position
    }

    /// Get the drift function of the Generalized Langevin equation
    pub fn drift_func(&self) -> &D {
        &self.drift_func
    }

    /// Get the diffusion function of the Generalized Langevin equation
    pub fn diffusion_func(&self) -> &G {
        &self.diffusion_func
    }

    /// Get the stability index of the Generalized Langevin equation
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Calculate the mean of the Generalized Langevin equation
    pub fn mean(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.raw_moment(1, particles, time_step)
    }

    /// Calculate the mean square displacement of the Generalized Langevin equation
    pub fn msd(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.central_moment(2, particles, time_step)
    }

    /// Calculate the raw moment of the Generalized Langevin equation
    pub fn raw_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.raw_moment(order, particles, time_step)
    }

    /// Calculate the central moment of the Generalized Langevin equation
    pub fn central_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.central_moment(order, particles, time_step)
    }

    /// Calculate the first passage time of the Generalized Langevin equation
    pub fn fpt(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        max_duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<Option<f64>> {
        let fpt = FirstPassageTime::new(self, domain)?;
        fpt.simulate(max_duration, time_step)
    }

    /// Calculate the occupation time of the Generialized Langevin equation
    pub fn occupation_time(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<f64> {
        let oc = OccupationTime::new(self, domain, duration)?;
        oc.simulate(time_step)
    }
}

/// impl `ContinuousProcess` trait for Generalized Langevin equation
impl<D, G> ContinuousProcess for GeneralizedLangevin<D, G>
where
    D: Fn(f64, f64) -> f64 + Clone + Send + Sync,
    G: Fn(f64, f64) -> f64 + Clone + Send + Sync,
{
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        simulate_generalized_langevin(
            &self.drift_func,
            &self.diffusion_func,
            self.start_position(),
            self.alpha,
            duration,
            time_step,
        )
    }
}

/// Simulate the Generalized Langevin equation
///
/// # Arguments
///
/// - `drift`: the drift function of the Generalized Langevin equation.
/// - `diffusion`: the diffusion function of the Generalized Langevin equation.
/// - `start_position`: the starting position of the Generalized Langevin equation.
/// - `alpha`: the stability index of the Generalized Langevin equation.
/// - `duration`: the duration of the simulation.
/// - `time_step`: the time step of the simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::generalized_langevin::simulate_generalized_langevin;
/// let drift = |x: f64, _t: f64| x;
/// let diffusion = |_x: f64, _t: f64| 1.0;
/// let start_position = 0.0;
/// let alpha = 1.7;
/// let duration = 1.0;
/// let time_step = 0.01;
/// let (t, x) = simulate_generalized_langevin(&drift, &diffusion, start_position, alpha, duration, time_step).unwrap();
/// ```
pub fn simulate_generalized_langevin<D, G>(
    drift: &D,
    diffusion: &G,
    start_position: impl Into<f64>,
    alpha: impl Into<f64>,
    duration: impl Into<f64>,
    time_step: f64,
) -> XResult<Pair>
where
    D: Fn(f64, f64) -> f64 + Send + Sync,
    G: Fn(f64, f64) -> f64 + Send + Sync,
{
    let duration = duration.into();
    let alpha = alpha.into();
    let num = (duration / time_step).ceil() as usize;
    let t = (0..=num)
        .into_par_iter()
        .map(|i| time_step * i as f64)
        .collect::<Vec<_>>();
    let mut x = vec![0.0; num + 1];
    x[0] = start_position.into();
    let noise = stable::sym_standard_rands(alpha, num)?;
    for i in 1..=num {
        x[i] = x[i - 1]
            + drift(x[i - 1], t[i - 1]) * time_step
            + diffusion(x[i - 1], t[i - 1]) * noise[i - 1] * time_step.powf(1.0 / alpha);
    }
    Ok((t, x))
}

/// Subordinated Langevin equation
///
/// dx(t) = f(x(t), t) dS(t) + g(x(t), t) dB(S(t)), x(0) = x0
///
/// where S(t) is the `alpha`-stable subordinator.
///
/// # Fields
///
/// - `drift_func`: the drift function of the Generalized Langevin equation, f(x, t).
/// - `diffusion_func`: the diffusion function of the Generalized Langevin equation, g(x, t).
/// - `start_position`: the starting position of the Generalized Langevin equation, x0.
/// - `alpha`: the stability index of the alpha-stable subordinator.
#[derive(Clone)]
pub struct SubordinatedLangevin<D, G>
where
    D: Fn(f64, f64) -> f64 + Clone + Send + Sync,
    G: Fn(f64, f64) -> f64 + Clone + Send + Sync,
{
    drift_func: D,
    diffusion_func: G,
    start_position: f64,
    alpha: f64,
}

impl<D, G> SubordinatedLangevin<D, G>
where
    D: Fn(f64, f64) -> f64 + Clone + Send + Sync,
    G: Fn(f64, f64) -> f64 + Clone + Send + Sync,
{
    /// Create a new SubordinatedLangevin
    ///
    /// # Arguments
    ///
    /// - `drift_func`: the drift function of the SubordinatedLangevin.
    /// - `diffusion_func`: the diffusion function of the SubordinatedLangevin.
    /// - `start_position`: the starting position of the SubordinatedLangevin.
    /// - `alpha`: the stability index of the alpha-stable subordinator.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::SubordinatedLangevin, prelude::*};
    /// let drift = |x: f64, _t: f64| x;
    /// let diffusion = |_x: f64, _t: f64| 1.0;
    /// let start_position = 0.0;
    /// let alpha = 0.5;
    /// let langevin = SubordinatedLangevin::new(drift, diffusion, start_position, alpha)
    ///     .unwrap();
    /// ```
    pub fn new(
        drift_func: D,
        diffusion_func: G,
        start_position: impl Into<f64>,
        alpha: impl Into<f64>,
    ) -> XResult<Self> {
        let start_position = start_position.into();
        let alpha = alpha.into();
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(SimulationError::InvalidParameters(format!(
                "The `alpha` must be in the range (0, 1), got {}",
                alpha
            ))
            .into());
        }
        Ok(Self {
            drift_func,
            diffusion_func,
            start_position,
            alpha,
        })
    }

    /// Simulate the SubordinatedLangevin
    ///
    /// # Arguments
    ///
    /// - `duration`: the duration of the simulation.
    /// - `time_step`: the time step of the simulation.
    pub fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        simulate_subordinated_langevin(
            &self.drift_func,
            &self.diffusion_func,
            self.start_position(),
            self.alpha,
            duration,
            time_step,
        )
    }

    /// Get the starting position of the SubordinatedLangevin
    pub fn start_position(&self) -> f64 {
        self.start_position
    }

    /// Get the drift function of the SubordinatedLangevin
    pub fn drift_func(&self) -> &D {
        &self.drift_func
    }

    /// Get the diffusion function of the SubordinatedLangevin
    pub fn diffusion_func(&self) -> &G {
        &self.diffusion_func
    }

    /// Get the stability index of the SubordinatedLangevin
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Calculate the mean of the SubordinatedLangevin
    pub fn mean(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.raw_moment(1, particles, time_step)
    }

    /// Calculate the mean square displacement of the SubordinatedLangevin
    pub fn msd(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.central_moment(2, particles, time_step)
    }

    /// Calculate the raw moment of the SubordinatedLangevin
    pub fn raw_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.raw_moment(order, particles, time_step)
    }

    /// Calculate the central moment of the SubordinatedLangevin
    pub fn central_moment(
        &self,
        duration: impl Into<f64>,
        order: i32,
        particles: usize,
        time_step: f64,
    ) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.central_moment(order, particles, time_step)
    }

    /// Calculate the first passage time of the SubordinatedLangevin
    pub fn fpt(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        max_duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<Option<f64>> {
        let fpt = FirstPassageTime::new(self, domain)?;
        fpt.simulate(max_duration, time_step)
    }

    /// Calculate the occupation time of the SubordinatedLangevin
    pub fn occupation_time(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<f64> {
        let oc = OccupationTime::new(self, domain, duration)?;
        oc.simulate(time_step)
    }
}

/// impl `ContinuousProcess` trait for SubordinatedLangevin
impl<D, G> ContinuousProcess for SubordinatedLangevin<D, G>
where
    D: Fn(f64, f64) -> f64 + Clone + Send + Sync,
    G: Fn(f64, f64) -> f64 + Clone + Send + Sync,
{
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        simulate_subordinated_langevin(
            &self.drift_func,
            &self.diffusion_func,
            self.start_position(),
            self.alpha,
            duration,
            time_step,
        )
    }
}

/// Simulate the subordinated Langevin equation
///
/// # Arguments
///
/// - `drift`: the drift function of the subordinated Langevin equation.
/// - `diffusion`: the diffusion function of the subordinated Langevin equation.
/// - `start_position`: the starting position of the subordinated Langevin equation.
/// - `alpha`: the stability index of the alpha-stable subordinator.
/// - `duration`: the duration of the simulation.
/// - `time_step`: the time step of the simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::subordinated_langevin::simulate_subordinated_langevin;
/// let drift = |x: f64, _t: f64| x;
/// let diffusion = |_x: f64, _t: f64| 1.0;
/// let start_position = 0.0;
/// let alpha = 0.5;
/// let duration = 1.0;
/// let time_step = 0.01;
/// let (t, x) = simulate_subordinated_langevin(&drift, &diffusion, start_position, alpha, duration, time_step).unwrap();
/// ```
pub fn simulate_subordinated_langevin<D, G>(
    drift: &D,
    diffusion: &G,
    start_position: impl Into<f64>,
    alpha: impl Into<f64>,
    duration: impl Into<f64>,
    time_step: f64,
) -> XResult<Pair>
where
    D: Fn(f64, f64) -> f64 + Send + Sync,
    G: Fn(f64, f64) -> f64 + Send + Sync,
{
    let duration = duration.into();
    let alpha = alpha.into();
    let num = (duration / time_step).ceil() as usize;
    let t = (0..=num)
        .into_par_iter()
        .map(|i| time_step * i as f64)
        .collect::<Vec<_>>();
    let mut x = vec![0.0; num + 1];
    x[0] = start_position.into();
    let (_, s) = Subordinator::new(alpha)?.simulate(duration, time_step)?;
    let noise = normal::standard_rands(num);
    for i in 1..=num {
        let delta_t = s[i] - s[i - 1];
        x[i] = x[i - 1]
            + drift(x[i - 1], t[i - 1]) * delta_t
            + diffusion(x[i - 1], t[i - 1]) * noise[i - 1] * delta_t.sqrt();
    }
    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_generalized_langevin() {
        let langevin = GeneralizedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 1.7)
            .expect("Failed to create GeneralizedLangevin");
        let (t, x) = langevin.simulate(1.0, 0.01).expect("Failed to simulate");
        assert_eq!(t.len(), x.len());
        assert!(t.last().expect("Empty time vector") <= &1.0);
    }

    #[test]
    fn test_with_generalized_closure() {
        let a = 2.0;
        let b = 3.0;
        // 使用捕获外部变量的闭包
        let langevin = GeneralizedLangevin::new(move |x, _t| a * x, move |_x, _t| b, 0.0, 1.7)
            .expect("Failed to create GeneralizedLangevin");
        let (t, x) = langevin.simulate(1.0, 0.01).expect("Failed to simulate");
        assert_eq!(t.len(), x.len());
    }

    #[test]
    fn test_mean() {
        let langevin = GeneralizedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 1.7)
            .expect("Failed to create GeneralizedLangevin");
        let _mean = langevin
            .mean(1.0, 1000, 0.01)
            .expect("Failed to calculate mean");
        // assert!(mean > 0.0);
    }

    #[test]
    fn test_msd() {
        let langevin = GeneralizedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 1.7)
            .expect("Failed to create GeneralizedLangevin");
        let msd = langevin
            .msd(1.0, 1000, 0.01)
            .expect("Failed to calculate msd");
        assert!(msd > 0.0);
    }

    #[test]
    fn test_raw_moment() {
        let langevin = GeneralizedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 1.7)
            .expect("Failed to create GeneralizedLangevin");
        let _raw_moment = langevin
            .raw_moment(1.0, 1, 1000, 0.01)
            .expect("Failed to calculate raw moment");
    }

    #[test]
    fn test_central_moment() {
        let langevin = GeneralizedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 1.7)
            .expect("Failed to create GeneralizedLangevin");
        let _central_moment = langevin
            .central_moment(1.0, 2, 1000, 0.01)
            .expect("Failed to calculate central moment");
    }

    #[test]
    fn test_fpt() {
        let langevin = GeneralizedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 1.7)
            .expect("Failed to create GeneralizedLangevin");
        let _fpt = langevin
            .fpt((0.0, 1.0), 1.0, 0.01)
            .expect("Failed to calculate fpt");
    }

    #[test]
    fn test_occupation_time() {
        let langevin = GeneralizedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 1.7)
            .expect("Failed to create GeneralizedLangevin");
        let _occupation_time = langevin
            .occupation_time((0.0, 10.0), 1.0, 0.01)
            .expect("Failed to calculate occupation time");
        // println!("occupation_time: {}", occupation_time);
        // assert!(occupation_time > 0.0);
    }

    #[test]
    fn test_simulate_subordinated_langevin() {
        let langevin = SubordinatedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 0.5)
            .expect("Failed to create SubordinatedLangevin");
        let (t, x) = langevin.simulate(1.0, 0.01).expect("Failed to simulate");
        assert_eq!(t.len(), x.len());
        assert!(t.last().expect("Empty time vector") <= &1.0);
    }

    #[test]
    fn test_with_subordinated_closure() {
        let a = 2.0;
        let b = 3.0;
        // 使用捕获外部变量的闭包
        let langevin = SubordinatedLangevin::new(move |x, _t| a * x, move |_x, _t| b, 0.0, 0.5)
            .expect("Failed to create SubordinatedLangevin");
        let (t, x) = langevin.simulate(1.0, 0.01).expect("Failed to simulate");
        assert_eq!(t.len(), x.len());
    }

    #[test]
    fn test_subordinated_langevin_mean() {
        let langevin = SubordinatedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 0.5)
            .expect("Failed to create SubordinatedLangevin");
        let _mean = langevin
            .mean(1.0, 1000, 0.01)
            .expect("Failed to calculate mean");
        // assert!(mean > 0.0);
    }
}
