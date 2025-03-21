//! Langevin equation simulation
//!

use crate::{XResult, random::normal, simulation::prelude::*};
use rayon::prelude::*;

/// Langevin equation
///
/// dx(t) = f(x(t), t) dt + g(x(t), t) dW(t), x(0) = x0
///
/// where W(t) is the Weiner process or called Brownian motion.
///
/// # Fields
///
/// - `drift_func`: the drift function of the Langevin equation, f(x, t).
/// - `diffusion_func`: the diffusion function of the Langevin equation, g(x, t).
/// - `start_position`: the starting position of the Langevin equation, x0.
#[derive(Clone)]
pub struct Langevin<D, G>
where
    D: Fn(f64, f64) -> f64 + Clone + Send + Sync,
    G: Fn(f64, f64) -> f64 + Clone + Send + Sync,
{
    drift_func: D,
    diffusion_func: G,
    start_position: f64,
}

impl<D, G> Langevin<D, G>
where
    D: Fn(f64, f64) -> f64 + Clone + Send + Sync,
    G: Fn(f64, f64) -> f64 + Clone + Send + Sync,
{
    pub fn new(drift_func: D, diffusion_func: G, start_position: impl Into<f64>) -> XResult<Self> {
        let start_position = start_position.into();
        Ok(Self {
            drift_func,
            diffusion_func,
            start_position,
        })
    }

    pub fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        simulate_langevin(
            &self.drift_func,
            &self.diffusion_func,
            self.start_position(),
            duration,
            time_step,
        )
    }

    pub fn start_position(&self) -> f64 {
        self.start_position
    }

    pub fn drift_func(&self) -> &D {
        &self.drift_func
    }

    pub fn diffusion_func(&self) -> &G {
        &self.diffusion_func
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

impl<D, G> ContinuousProcess for Langevin<D, G>
where
    D: Fn(f64, f64) -> f64 + Clone + Send + Sync,
    G: Fn(f64, f64) -> f64 + Clone + Send + Sync,
{
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        simulate_langevin(
            &self.drift_func,
            &self.diffusion_func,
            self.start_position(),
            duration,
            time_step,
        )
    }
}

pub fn simulate_langevin<D, G>(
    drift: &D,
    diffusion: &G,
    start_position: impl Into<f64>,
    duration: impl Into<f64>,
    time_step: f64,
) -> XResult<Pair>
where
    D: Fn(f64, f64) -> f64 + Send + Sync,
    G: Fn(f64, f64) -> f64 + Send + Sync,
{
    let duration = duration.into();
    let num = (duration / time_step).ceil() as usize;
    let t = (0..=num)
        .into_par_iter()
        .map(|i| time_step * i as f64)
        .collect::<Vec<_>>();
    let mut x = vec![0.0; num + 1];
    x[0] = start_position.into();
    let noise = normal::standard_rands(num);
    for i in 1..=num {
        x[i] = x[i - 1]
            + drift(x[i - 1], t[i - 1]) * time_step
            + diffusion(x[i - 1], t[i - 1]) * noise[i - 1] * time_step.sqrt();
    }
    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_langevin() {
        let langevin =
            Langevin::new(|x, _t| x, |_x, _t| 1.0, 0.0).expect("Failed to create Langevin");
        let (t, x) = langevin.simulate(1.0, 0.01).expect("Failed to simulate");
        assert_eq!(t.len(), x.len());
        assert!(t.last().expect("Empty time vector") <= &1.0);
    }

    #[test]
    fn test_with_closure() {
        let a = 2.0;
        let b = 3.0;
        // 使用捕获外部变量的闭包
        let langevin = Langevin::new(move |x, _t| a * x, move |_x, _t| b, 0.0)
            .expect("Failed to create Langevin");
        let (t, x) = langevin.simulate(1.0, 0.01).expect("Failed to simulate");
        assert_eq!(t.len(), x.len());
    }

    #[test]
    fn test_mean() {
        let langevin =
            Langevin::new(|x, _t| x, |_x, _t| 1.0, 0.0).expect("Failed to create Langevin");
        let _mean = langevin
            .mean(1.0, 1000, 0.01)
            .expect("Failed to calculate mean");
        // assert!(mean > 0.0);
    }

    #[test]
    fn test_msd() {
        let langevin =
            Langevin::new(|x, _t| x, |_x, _t| 1.0, 0.0).expect("Failed to create Langevin");
        let msd = langevin
            .msd(1.0, 1000, 0.01)
            .expect("Failed to calculate msd");
        assert!(msd > 0.0);
    }

    #[test]
    fn test_raw_moment() {
        let langevin =
            Langevin::new(|x, _t| x, |_x, _t| 1.0, 0.0).expect("Failed to create Langevin");
        let _raw_moment = langevin
            .raw_moment(1.0, 1, 1000, 0.01)
            .expect("Failed to calculate raw moment");
        // assert!(raw_moment > 0.0);
    }

    #[test]
    fn test_central_moment() {
        let langevin =
            Langevin::new(|x, _t| x, |_x, _t| 1.0, 0.0).expect("Failed to create Langevin");
        let central_moment = langevin
            .central_moment(1.0, 2, 1000, 0.01)
            .expect("Failed to calculate central moment");
        assert!(central_moment > 0.0);
    }

    #[test]
    fn test_fpt() {
        let langevin =
            Langevin::new(|x, _t| x, |_x, _t| 1.0, 0.0).expect("Failed to create Langevin");
        let fpt = langevin
            .fpt((0.0, 1.0), 1.0, 0.01)
            .expect("Failed to calculate fpt");
        assert!(fpt.is_some());
    }

    #[test]
    fn test_occupation_time() {
        let langevin =
            Langevin::new(|x, _t| x, |_x, _t| 1.0, 0.0).expect("Failed to create Langevin");
        let occupation_time = langevin
            .occupation_time((0.0, 10.0), 1.0, 0.01)
            .expect("Failed to calculate occupation time");
        println!("occupation_time: {}", occupation_time);
        // assert!(occupation_time > 0.0);
    }
}
