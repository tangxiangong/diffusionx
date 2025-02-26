//! Generalized Langevin equation and subordinated Langevin equation simulation
//!

use crate::{
    SimulationError, XResult,
    random::{normal, stable},
    simulation::{Subordinator, prelude::*},
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
pub struct GeneralizedLangevin {
    drift_func: fn(f64, f64) -> f64,
    diffusion_func: fn(f64, f64) -> f64,
    start_position: f64,
    alpha: f64,
}

impl GeneralizedLangevin {
    pub fn new(
        drift_func: fn(f64, f64) -> f64,
        diffusion_func: fn(f64, f64) -> f64,
        start_position: impl Into<f64>,
        alpha: impl Into<f64>,
    ) -> XResult<Self> {
        let start_position = start_position.into();
        let alpha = alpha.into();
        if alpha <= 0.0 || alpha > 2.0 {
            return Err(SimulationError::InvalidParameters(
                "alpha must be in the range (0, 2]".to_string(),
            )
            .into());
        }
        Ok(Self {
            drift_func,
            diffusion_func,
            start_position,
            alpha,
        })
    }

    pub fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        simulate_generalized_langevin(
            self.drift_func(),
            self.diffusion_func(),
            self.start_position(),
            self.alpha,
            duration,
            time_step,
        )
    }

    pub fn start_position(&self) -> f64 {
        self.start_position
    }

    pub fn drift_func(&self) -> fn(f64, f64) -> f64 {
        self.drift_func
    }

    pub fn diffusion_func(&self) -> fn(f64, f64) -> f64 {
        self.diffusion_func
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn mean(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.raw_moment(1, particles, time_step)
    }

    pub fn msd(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.central_moment(2, particles, time_step)
    }

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

    pub fn fpt(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        max_duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<Option<f64>> {
        let fpt = FirstPassageTime::new(self, domain)?;
        fpt.simulate(max_duration, time_step)
    }

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

impl ContinuousProcess for GeneralizedLangevin {
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        simulate_generalized_langevin(
            self.drift_func(),
            self.diffusion_func(),
            self.start_position(),
            self.alpha,
            duration,
            time_step,
        )
    }
}

pub fn simulate_generalized_langevin(
    drift: fn(f64, f64) -> f64,
    diffusion: fn(f64, f64) -> f64,
    start_position: impl Into<f64>,
    alpha: impl Into<f64>,
    duration: impl Into<f64>,
    time_step: f64,
) -> XResult<Pair> {
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
pub struct SubordinatedLangevin {
    drift_func: fn(f64, f64) -> f64,
    diffusion_func: fn(f64, f64) -> f64,
    start_position: f64,
    alpha: f64,
}

impl SubordinatedLangevin {
    pub fn new(
        drift_func: fn(f64, f64) -> f64,
        diffusion_func: fn(f64, f64) -> f64,
        start_position: impl Into<f64>,
        alpha: impl Into<f64>,
    ) -> XResult<Self> {
        let start_position = start_position.into();
        let alpha = alpha.into();
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(SimulationError::InvalidParameters(
                "alpha must be in the range (0, 1)".to_string(),
            )
            .into());
        }
        Ok(Self {
            drift_func,
            diffusion_func,
            start_position,
            alpha,
        })
    }

    pub fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        simulate_subordinated_langevin(
            self.drift_func(),
            self.diffusion_func(),
            self.start_position(),
            self.alpha,
            duration,
            time_step,
        )
    }

    pub fn start_position(&self) -> f64 {
        self.start_position
    }

    pub fn drift_func(&self) -> fn(f64, f64) -> f64 {
        self.drift_func
    }

    pub fn diffusion_func(&self) -> fn(f64, f64) -> f64 {
        self.diffusion_func
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn mean(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.raw_moment(1, particles, time_step)
    }

    pub fn msd(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64> {
        let traj = self.duration(duration)?;
        traj.central_moment(2, particles, time_step)
    }

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

    pub fn fpt(
        &self,
        domain: (impl Into<f64>, impl Into<f64>),
        max_duration: impl Into<f64>,
        time_step: f64,
    ) -> XResult<Option<f64>> {
        let fpt = FirstPassageTime::new(self, domain)?;
        fpt.simulate(max_duration, time_step)
    }

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

impl ContinuousProcess for SubordinatedLangevin {
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        simulate_subordinated_langevin(
            self.drift_func(),
            self.diffusion_func(),
            self.start_position(),
            self.alpha,
            duration,
            time_step,
        )
    }
}

pub fn simulate_subordinated_langevin(
    drift: fn(f64, f64) -> f64,
    diffusion: fn(f64, f64) -> f64,
    start_position: impl Into<f64>,
    alpha: impl Into<f64>,
    duration: impl Into<f64>,
    time_step: f64,
) -> XResult<Pair> {
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
        let langevin = GeneralizedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 1.7).unwrap();
        let (t, x) = langevin.simulate(1.0, 0.01).unwrap();
        assert_eq!(t.len(), x.len());
        assert!(t.last().unwrap() <= &1.0);
    }

    #[test]
    fn test_mean() {
        let langevin = GeneralizedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 1.7).unwrap();
        let mean = langevin.mean(1.0, 1000, 0.01).unwrap();
        assert!(mean > 0.0);
    }

    #[test]
    fn test_msd() {
        let langevin = GeneralizedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 1.7).unwrap();
        let msd = langevin.msd(1.0, 1000, 0.01).unwrap();
        assert!(msd > 0.0);
    }

    #[test]
    fn test_raw_moment() {
        let langevin = GeneralizedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 1.7).unwrap();
        let raw_moment = langevin.raw_moment(1.0, 1, 1000, 0.01).unwrap();
        assert!(raw_moment > 0.0);
    }

    #[test]
    fn test_central_moment() {
        let langevin = GeneralizedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 1.7).unwrap();
        let central_moment = langevin.central_moment(1.0, 2, 1000, 0.01).unwrap();
        assert!(central_moment > 0.0);
    }

    #[test]
    fn test_fpt() {
        let langevin = GeneralizedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 1.7).unwrap();
        let fpt = langevin.fpt((0.0, 1.0), 1.0, 0.01).unwrap();
        assert!(fpt.is_some());
    }

    #[test]
    fn test_occupation_time() {
        let langevin = GeneralizedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 1.7).unwrap();
        let occupation_time = langevin.occupation_time((0.0, 10.0), 1.0, 0.01).unwrap();
        println!("occupation_time: {}", occupation_time);
        // assert!(occupation_time > 0.0);
    }

    #[test]
    fn test_simulate_subordinated_langevin() {
        let langevin = SubordinatedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 0.5).unwrap();
        let (t, x) = langevin.simulate(1.0, 0.01).unwrap();
        assert_eq!(t.len(), x.len());
        assert!(t.last().unwrap() <= &1.0);
    }

    #[test]
    fn test_subordinated_langevin_mean() {
        let langevin = SubordinatedLangevin::new(|x, _t| x, |_x, _t| 1.0, 0.0, 0.5).unwrap();
        let mean = langevin.mean(1.0, 1000, 0.01).unwrap();
        assert!(mean > 0.0);
    }
}
