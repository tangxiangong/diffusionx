//! Langevin equation simulation

use crate::{XResult, random::normal, simulation::prelude::*};

/// Langevin equation
///
/// $$dx(t) = f(x(t), t) dt + g(x(t), t) dW(t),\qquad x(0) = x_0$$
///
/// where $W(t)$ is the Weiner process or called Brownian motion.
#[derive(Debug, Clone)]
pub struct Langevin<D, G>
where
    D: Fn(f64, f64) -> f64 + Clone + Send + Sync,
    G: Fn(f64, f64) -> f64 + Clone + Send + Sync,
{
    /// The drift function
    drift_func: D,
    /// The diffusion function
    diffusion_func: G,
    /// The starting position
    start_position: f64,
}

impl<D, G> Langevin<D, G>
where
    D: Fn(f64, f64) -> f64 + Clone + Send + Sync,
    G: Fn(f64, f64) -> f64 + Clone + Send + Sync,
{
    /// Create a new `Langevin`
    ///
    /// # Arguments
    ///
    /// * `drift_func` - The drift function.
    /// * `diffusion_func` - The diffusion function.
    /// * `start_position` - The starting position.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::Langevin;
    ///
    /// let drift = |x: f64, _t: f64| x;
    /// let diffusion = |_x: f64, _t: f64| 1.0;
    /// let start_position = 0.0;
    /// let langevin = Langevin::new(drift, diffusion, start_position).unwrap();
    /// ```
    pub fn new(drift_func: D, diffusion_func: G, start_position: impl Into<f64>) -> XResult<Self> {
        let start_position = start_position.into();
        Ok(Self {
            drift_func,
            diffusion_func,
            start_position,
        })
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> f64 {
        self.start_position
    }

    /// Get the drift function
    pub fn get_drift_func(&self) -> &D {
        &self.drift_func
    }

    /// Get the diffusion function
    pub fn get_diffusion_func(&self) -> &G {
        &self.diffusion_func
    }
}

impl<D, G> ContinuousProcess for Langevin<D, G>
where
    D: Fn(f64, f64) -> f64 + Clone + Send + Sync,
    G: Fn(f64, f64) -> f64 + Clone + Send + Sync,
{
    fn start(&self) -> f64 {
        self.start_position
    }

    fn simulate_unchecked(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_langevin(
            &self.drift_func,
            &self.diffusion_func,
            self.start_position,
            duration,
            time_step,
        )
    }

    fn displacement(&self, duration: f64, time_step: f64) -> XResult<f64> {
        let num_steps = (duration / time_step).ceil() as usize;
        let noise = normal::standard_rands::<f64>(num_steps);
        let sigma = time_step.sqrt();
        let mut current_x = self.start_position;

        for i in 0..num_steps - 1 {
            let current_t = i as f64 * time_step;
            let xi = unsafe { *noise.get_unchecked(i) };

            current_x += self.get_drift_func()(current_x, current_t) * time_step
                + self.get_diffusion_func()(current_x, current_t) * xi * sigma;
        }

        let current_t = (num_steps - 1) as f64 * time_step;
        let last_step = duration - current_t;

        let xi = unsafe { *noise.get_unchecked(num_steps - 1) };
        current_x += self.get_drift_func()(current_x, current_t) * last_step
            + self.get_diffusion_func()(current_x, current_t) * xi * sigma;

        Ok(current_x - self.start_position)
    }
}

/// Simulate the Langevin equation
///
/// # Arguments
///
/// * `drift` - The drift function.
/// * `diffusion` - The diffusion function.
/// * `start_position` - The starting position.
/// * `duration` - The duration.
/// * `time_step` - The time step.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::langevin::simulate_langevin;
///
/// let drift = |x: f64, _t: f64| x;
/// let diffusion = |_x: f64, _t: f64| 1.0;
/// let start_position = 0.0;
/// let (t, x) = simulate_langevin(&drift, &diffusion, start_position, 1.0, 0.01).unwrap();
/// ```
pub fn simulate_langevin<D, G>(
    drift: &D,
    diffusion: &G,
    start_position: f64,
    duration: f64,
    time_step: f64,
) -> XResult<Pair>
where
    D: Fn(f64, f64) -> f64 + Send + Sync,
    G: Fn(f64, f64) -> f64 + Send + Sync,
{
    let num_steps = (duration / time_step).ceil() as usize;

    let mut t = Vec::with_capacity(num_steps + 1);
    let mut x = Vec::with_capacity(num_steps + 1);

    t.push(0.0);
    x.push(start_position);

    let noise = normal::standard_rands::<f64>(num_steps - 1);
    let sigma = time_step.sqrt();

    let mut current_t = 0.0;
    let mut current_x = start_position;
    for xi in noise {
        current_x +=
            drift(current_x, current_t) * time_step + diffusion(current_x, current_t) * xi * sigma;
        x.push(current_x);
        current_t += time_step;
        t.push(current_t);
    }

    let last_step = duration - current_t;
    let sigma = last_step.sqrt();
    let xi = normal::standard_rand::<f64>();
    current_x +=
        drift(current_x, current_t) * last_step + diffusion(current_x, current_t) * xi * sigma;
    x.push(current_x);
    t.push(duration);

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
        println!("occupation_time: {occupation_time}");
        // assert!(occupation_time > 0.0);
    }
}
