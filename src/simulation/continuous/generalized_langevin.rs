//! Generalized Langevin equation and subordinated Langevin equation simulation

use crate::{
    FloatExt, SimulationError, XResult, check_duration_time_step,
    random::{normal, stable},
    simulation::{continuous::Subordinator, prelude::*},
};
use num_traits::FloatConst;
use rand_distr::{Distribution, Exp1, StandardNormal, uniform::SampleUniform};

/// Generalized Langevin equation
///
/// $$dx(t) = f(x(t), t) dt + g(x(t), t) dL_\alpha(t),\qquad x(0) = x_0$$
///
/// where $L_\alpha(t)$ is the $\alpha$-stable process.
#[derive(Debug, Clone)]
pub struct GeneralizedLangevin<D, G, T: FloatExt = f64>
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
    /// The stability index
    alpha: T,
}

impl<D, G, T: FloatExt> GeneralizedLangevin<D, G, T>
where
    D: Fn(T, T) -> T + Clone + Send + Sync,
    G: Fn(T, T) -> T + Clone + Send + Sync,
{
    /// Create a new `GeneralizedLangevin`
    ///
    /// # Arguments
    ///
    /// * `drift_func` - The drift function.
    /// * `diffusion_func` - The diffusion function.
    /// * `start_position` - The starting position.
    /// * `alpha` - The stability index.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::GeneralizedLangevin;
    ///
    /// let drift_func = |x: f64, _t: f64| x;
    /// let diffusion_func = |_x: f64, _t: f64| 1.0;
    /// let start_position = 0.0;
    /// let alpha = 1.7;
    /// let langevin =
    ///     GeneralizedLangevin::new(drift_func, diffusion_func, start_position, alpha).unwrap();
    /// ```
    pub fn new(drift_func: D, diffusion_func: G, start_position: T, alpha: T) -> XResult<Self> {
        if alpha <= T::zero() || alpha > T::from(2).unwrap() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `alpha` must be in the range (0, 2], got {alpha:?}"
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

    /// Get the stability index
    pub fn get_alpha(&self) -> T {
        self.alpha
    }
}

impl<D, G, T: FloatExt + FloatConst + SampleUniform> ContinuousProcess<T>
    for GeneralizedLangevin<D, G, T>
where
    D: Fn(T, T) -> T + Clone + Send + Sync,
    G: Fn(T, T) -> T + Clone + Send + Sync,
    Exp1: Distribution<T>,
{
    fn start(&self) -> T {
        self.start_position
    }

    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        simulate_generalized_langevin(
            &self.drift_func,
            &self.diffusion_func,
            self.start_position,
            self.alpha,
            duration,
            time_step,
        )
    }

    fn displacement(&self, duration: T, time_step: T) -> XResult<T> {
        check_duration_time_step(duration, time_step)?;

        let drift = self.get_drift_func();
        let diffusion = self.get_diffusion_func();
        let num_steps = (duration / time_step).ceil().to_usize().unwrap();
        let mut scale = time_step.powf(T::one() / self.alpha);

        let mut current_t = T::zero();
        let mut current_x = self.start_position;
        let mut mu;
        let mut diffusivity;

        let noises = stable::sym_standard_rands(self.alpha, num_steps - 1)?;

        for xi in noises {
            mu = drift(current_x, current_t);
            diffusivity = diffusion(current_x, current_t);
            current_x += mu * time_step + diffusivity * xi * scale;
            current_t += time_step;
        }
        let last_step = duration - current_t;
        scale = last_step.powf(T::one() / self.alpha);
        mu = drift(current_x, current_t);
        diffusivity = diffusion(current_x, current_t);
        current_x += mu * last_step + diffusivity * stable::sym_standard_rand(self.alpha)? * scale;
        Ok(current_x - self.start_position)
    }
}

/// Simulate the Generalized Langevin equation
///
/// # Arguments
///
/// - `drift` - The drift function.
/// - `diffusion` - The diffusion function.
/// - `start_position` - The starting position.
/// - `alpha` - The stability index.
/// - `duration` - The duration of the simulation.
/// - `time_step` - The time step of the simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::generalized_langevin::simulate_generalized_langevin;
///
/// let drift = |x: f64, _t: f64| x;
/// let diffusion = |_x: f64, _t: f64| 1.0;
/// let start_position = 0.0;
/// let alpha = 1.7;
/// let duration = 1.0;
/// let time_step = 0.01;
/// let (t, x) = simulate_generalized_langevin(
///     &drift,
///     &diffusion,
///     start_position,
///     alpha,
///     duration,
///     time_step,
/// )
/// .unwrap();
/// ```
pub fn simulate_generalized_langevin<D, G, T: FloatExt + FloatConst + SampleUniform>(
    drift: &D,
    diffusion: &G,
    start_position: T,
    alpha: T,
    duration: T,
    time_step: T,
) -> XResult<(Vec<T>, Vec<T>)>
where
    D: Fn(T, T) -> T + Send + Sync,
    G: Fn(T, T) -> T + Send + Sync,
    Exp1: Distribution<T>,
{
    check_duration_time_step(duration, time_step)?;

    let num_steps = (duration / time_step).ceil().to_usize().unwrap();

    let mut t = Vec::with_capacity(num_steps + 1);
    let mut x = Vec::with_capacity(num_steps + 1);

    t.push(T::zero());
    x.push(start_position);

    let mut scale = time_step.powf(T::one() / alpha);

    let mut current_x = start_position;
    let mut current_t = T::zero();
    let mut mu;
    let mut diffusivity;

    let noises = stable::sym_standard_rands(alpha, num_steps - 1)?;

    for xi in noises {
        mu = drift(current_x, current_t);
        diffusivity = diffusion(current_x, current_t);
        current_x += mu * time_step + diffusivity * xi * scale;
        x.push(current_x);
        current_t += time_step;
        t.push(current_t);
    }

    let last_step = duration - current_t;
    scale = last_step.powf(T::one() / alpha);
    mu = drift(current_x, current_t);
    diffusivity = diffusion(current_x, current_t);
    current_x += mu * last_step + diffusivity * stable::sym_standard_rand(alpha)? * scale;
    t.push(duration);
    x.push(current_x);
    Ok((t, x))
}

/// Subordinated Langevin equation
///
/// dx(t) = f(x(t), t) dS(t) + g(x(t), t) dB(S(t)), x(0) = x0
///
/// where S(t) is the `alpha`-stable subordinator.
#[derive(Debug, Clone)]
pub struct SubordinatedLangevin<D, G, T: FloatExt = f64>
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
    /// The stability index
    alpha: T,
}

impl<D, G, T: FloatExt> SubordinatedLangevin<D, G, T>
where
    D: Fn(T, T) -> T + Clone + Send + Sync,
    G: Fn(T, T) -> T + Clone + Send + Sync,
{
    /// Create a new `SubordinatedLangevin`
    ///
    /// # Arguments
    ///
    /// - `drift_func` - The drift function.
    /// - `diffusion_func` - The diffusion function.
    /// - `start_position` - The starting position.
    /// - `alpha` - The stability index.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::{continuous::SubordinatedLangevin, prelude::*};
    ///
    /// let drift = |x: f64, _t: f64| x;
    /// let diffusion = |_x: f64, _t: f64| 1.0;
    /// let start_position = 0.0;
    /// let alpha = 0.5;
    /// let langevin = SubordinatedLangevin::new(drift, diffusion, start_position, alpha).unwrap();
    /// ```
    pub fn new(drift_func: D, diffusion_func: G, start_position: T, alpha: T) -> XResult<Self> {
        if alpha <= T::zero() || alpha >= T::one() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `alpha` must be in the range (0, 1), got {alpha:?}"
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

    /// Get the stability index
    pub fn get_alpha(&self) -> T {
        self.alpha
    }
}

impl<D, G, T: FloatExt + FloatConst + SampleUniform> ContinuousProcess<T>
    for SubordinatedLangevin<D, G, T>
where
    D: Fn(T, T) -> T + Clone + Send + Sync,
    G: Fn(T, T) -> T + Clone + Send + Sync,
    Exp1: Distribution<T>,
    StandardNormal: Distribution<T>,
{
    fn start(&self) -> T {
        self.start_position
    }

    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        simulate_subordinated_langevin(
            &self.drift_func,
            &self.diffusion_func,
            self.start_position,
            self.alpha,
            duration,
            time_step,
        )
    }

    fn displacement(&self, duration: T, time_step: T) -> XResult<T> {
        check_duration_time_step(duration, time_step)?;

        let (t, s) = Subordinator::new(self.alpha)?.simulate(duration, time_step)?;
        let num_steps = t.len() - 1;

        let drift = self.get_drift_func();
        let diffusion = self.get_diffusion_func();

        let mut current_x = self.start_position;
        let mut mu;
        let mut diffusivity;
        let mut si;
        let mut si_next;
        let mut delta_s;

        let noises = normal::standard_rands::<T>(num_steps);

        for ((&ti, sis), xi) in t.iter().zip(s.windows(2)).take(num_steps).zip(noises) {
            si = unsafe { *sis.get_unchecked(0) };
            si_next = unsafe { *sis.get_unchecked(1) };
            delta_s = si_next - si;
            mu = drift(current_x, ti);
            diffusivity = diffusion(current_x, ti);
            current_x += mu * delta_s + diffusivity * xi * delta_s.sqrt();
        }

        Ok(current_x - self.start_position)
    }
}

/// Simulate the subordinated Langevin equation
///
/// # Arguments
///
/// - `drift` - The drift function.
/// - `diffusion` - The diffusion function.
/// - `start_position` - The starting position.
/// - `alpha` - The stability index.
/// - `duration` - The duration of the simulation.
/// - `time_step` - The time step of the simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::subordinated_langevin::simulate_subordinated_langevin;
///
/// let drift = |x: f64, _t: f64| x;
/// let diffusion = |_x: f64, _t: f64| 1.0;
/// let start_position = 0.0;
/// let alpha = 0.5;
/// let duration = 1.0;
/// let time_step = 0.01;
/// let (t, x) = simulate_subordinated_langevin(
///     &drift,
///     &diffusion,
///     start_position,
///     alpha,
///     duration,
///     time_step,
/// )
/// .unwrap();
/// ```
pub fn simulate_subordinated_langevin<D, G, T: FloatExt + FloatConst + SampleUniform>(
    drift: &D,
    diffusion: &G,
    start_position: T,
    alpha: T,
    duration: T,
    time_step: T,
) -> XResult<(Vec<T>, Vec<T>)>
where
    D: Fn(T, T) -> T + Send + Sync,
    G: Fn(T, T) -> T + Send + Sync,
    Exp1: Distribution<T>,
    StandardNormal: Distribution<T>,
{
    check_duration_time_step(duration, time_step)?;

    let (t, s) = Subordinator::new(alpha)?.simulate(duration, time_step)?;
    let num_steps = t.len() - 1;

    let mut x = Vec::with_capacity(num_steps + 1);
    x.push(start_position);

    let mut mu;
    let mut diffusivity;
    let mut si;
    let mut si_next;
    let mut delta_s;
    let mut current_x = start_position;

    let noises = normal::standard_rands::<T>(num_steps);

    for ((&ti, sis), xi) in t.iter().zip(s.windows(2)).take(num_steps).zip(noises) {
        si = unsafe { *sis.get_unchecked(0) };
        si_next = unsafe { *sis.get_unchecked(1) };
        delta_s = si_next - si;
        mu = drift(current_x, ti);
        diffusivity = diffusion(current_x, ti);

        current_x += mu * delta_s + diffusivity * xi * delta_s.sqrt();

        x.push(current_x);
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
