//! Langevin equation simulation

use crate::{FloatExt, XResult, check_duration_time_step, random::normal, simulation::prelude::*};
use rand_distr::{Distribution, StandardNormal};

/// Langevin equation.
///
/// $$dX(t) = f(X(t), t)\,dt + g(X(t), t)\,dW(t),\qquad X(0)=x_0.$$
///
/// where \(W(t)\) is the Wiener process, also called Brownian motion.
#[derive(Debug, Clone)]
pub struct Langevin<D, G, T: FloatExt = f64>
where
    D: Fn(T, T) -> T + Clone + Send + Sync,
    G: Fn(T, T) -> T + Clone + Send + Sync,
{
    /// The drift function
    drift_func: D,
    /// The diffusion function
    diffusion_func: G,
    /// The starting position
    start_position: T,
}

impl<D, G, T: FloatExt> Langevin<D, G, T>
where
    D: Fn(T, T) -> T + Clone + Send + Sync,
    G: Fn(T, T) -> T + Clone + Send + Sync,
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
    pub fn new(drift_func: D, diffusion_func: G, start_position: T) -> XResult<Self> {
        Ok(Self {
            drift_func,
            diffusion_func,
            start_position,
        })
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> T {
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

impl<D, G, T: FloatExt> ContinuousProcess<T> for Langevin<D, G, T>
where
    D: Fn(T, T) -> T + Clone + Send + Sync,
    G: Fn(T, T) -> T + Clone + Send + Sync,
    StandardNormal: Distribution<T>,
{
    fn start(&self) -> T {
        self.start_position
    }

    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        simulate_langevin(
            &self.drift_func,
            &self.diffusion_func,
            self.start_position,
            duration,
            time_step,
        )
    }

    fn displacement(&self, duration: T, time_step: T) -> XResult<T> {
        check_duration_time_step(duration, time_step)?;

        let drift = self.get_drift_func();
        let diffusion = self.get_diffusion_func();

        let num_steps = (duration / time_step).ceil().to_usize().unwrap();

        let mut scale = time_step.sqrt();
        let mut current_x = self.start_position;
        let mut current_t = T::zero();
        let mut mu;
        let mut diffusivity;

        let noises = normal::standard_rands::<T>(num_steps - 1);

        for xi in noises {
            mu = drift(current_x, current_t);
            diffusivity = diffusion(current_x, current_t);
            current_x += mu * time_step + diffusivity * xi * scale;
            current_t += time_step;
        }

        let last_step = duration - current_t;
        scale = last_step.sqrt();
        mu = drift(current_x, current_t);
        diffusivity = diffusion(current_x, current_t);

        current_x += mu * last_step + diffusivity * normal::standard_rand::<T>() * scale;

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
pub fn simulate_langevin<D, G, T: FloatExt>(
    drift: &D,
    diffusion: &G,
    start_position: T,
    duration: T,
    time_step: T,
) -> XResult<(Vec<T>, Vec<T>)>
where
    D: Fn(T, T) -> T + Send + Sync,
    G: Fn(T, T) -> T + Send + Sync,
    StandardNormal: Distribution<T>,
{
    check_duration_time_step(duration, time_step)?;

    let num_steps = (duration / time_step).ceil().to_usize().unwrap();
    let mut t = Vec::with_capacity(num_steps + 1);
    let mut x = Vec::with_capacity(num_steps + 1);

    t.push(T::zero());
    x.push(start_position);

    let mut scale = time_step.sqrt();
    let noises = normal::standard_rands::<T>(num_steps - 1);
    let mut current_t = T::zero();
    let mut current_x = start_position;
    let mut mu;
    let mut diffusivity;

    for xi in noises {
        mu = drift(current_x, current_t);
        diffusivity = diffusion(current_x, current_t);
        current_x += mu * time_step + diffusivity * xi * scale;
        current_t += time_step;
        t.push(current_t);
        x.push(current_x);
    }

    let last_step = duration - current_t;
    scale = last_step.sqrt();
    mu = drift(current_x, current_t);
    diffusivity = diffusion(current_x, current_t);
    current_x += mu * last_step + diffusivity * normal::standard_rand::<T>() * scale;
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
