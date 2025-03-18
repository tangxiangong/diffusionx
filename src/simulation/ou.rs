//! Ornstein-Uhlenbeck process simulation
//!

use crate::{XResult, random::normal, simulation::prelude::*};
use rayon::prelude::*;

/// Ornstein–Uhlenbeck process
///
/// dx(t) = -theta x(t) dt + sigma dW(t), x(0) = x0
///
/// where W(t) is the Wiener process, also called Brownian motion.
///
/// # Fields
///
/// - `theta`: the parameter controlling the strength of mean reversion.
/// - `sigma`: the diffusion coefficient controlling the noise intensity.
/// - `start_position`: the initial position x0 of the process.
#[derive(Clone)]
pub struct OrnsteinUhlenbeck {
    theta: f64,
    sigma: f64,
    start_position: f64,
}

impl OrnsteinUhlenbeck {
    pub fn new(theta: f64, sigma: f64, start_position: impl Into<f64>) -> XResult<Self> {
        let start_position = start_position.into();
        Ok(Self {
            theta,
            sigma,
            start_position,
        })
    }

    pub fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        simulate_ou(
            self.theta,
            self.sigma,
            self.start_position(),
            duration,
            time_step,
        )
    }

    pub fn start_position(&self) -> f64 {
        self.start_position
    }

    pub fn theta(&self) -> f64 {
        self.theta
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
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

impl ContinuousProcess for OrnsteinUhlenbeck {
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        simulate_ou(
            self.theta,
            self.sigma,
            self.start_position(),
            duration,
            time_step,
        )
    }
}

pub fn simulate_ou(
    theta: f64,
    sigma: f64,
    start_position: impl Into<f64>,
    duration: impl Into<f64>,
    time_step: f64,
) -> XResult<Pair> {
    // 直接实现OU过程的数值模拟
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
        // OU特定的更新方程
        x[i] = x[i - 1] - theta * x[i - 1] * time_step + sigma * noise[i - 1] * time_step.sqrt();
    }

    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_ou() {
        let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
        let (t, x) = ou.simulate(1.0, 0.01).unwrap();
        assert_eq!(t.len(), x.len());
        assert!(t.last().unwrap() <= &1.0);
    }

    #[test]
    fn test_mean() {
        let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
        let _mean = ou.mean(1.0, 1000, 0.01).unwrap();
        // 由于随机过程的特性，这里不做具体数值断言
    }

    #[test]
    fn test_msd() {
        let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
        let msd = ou.msd(1.0, 1000, 0.01).unwrap();
        assert!(msd > 0.0);
    }

    #[test]
    fn test_raw_moment() {
        let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
        let _raw_moment = ou.raw_moment(1.0, 1, 1000, 0.01).unwrap();
        // 由于随机过程的特性，这里不做具体数值断言
    }

    #[test]
    fn test_central_moment() {
        let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
        let central_moment = ou.central_moment(1.0, 2, 1000, 0.01).unwrap();
        assert!(central_moment > 0.0);
    }

    #[test]
    fn test_fpt() {
        let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
        let _fpt = ou.fpt((-1.0, 1.0), 10.0, 0.01).unwrap();
    }

    #[test]
    fn test_occupation_time() {
        let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
        let occupation_time = ou.occupation_time((-1.0, 1.0), 1.0, 0.01).unwrap();
        assert!((0.0..=1.0).contains(&occupation_time));
    }
}
