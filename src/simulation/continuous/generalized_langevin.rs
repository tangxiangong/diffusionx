//! Generalized Langevin equation and subordinated Langevin equation simulation

use crate::{
    SimulationError, XResult,
    random::{normal, stable},
    simulation::{continuous::Subordinator, prelude::*},
};

/// Generalized Langevin equation
///
/// $$dx(t) = f(x(t), t) dt + g(x(t), t) dL_\alpha(t),\qquad x(0) = x_0$$
///
/// where $L_\alpha(t)$ is the $\alpha$-stable process.
#[derive(Debug, Clone)]
pub struct GeneralizedLangevin<D, G>
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
    /// The stability index
    alpha: f64,
}

impl<D, G> GeneralizedLangevin<D, G>
where
    D: Fn(f64, f64) -> f64 + Clone + Send + Sync,
    G: Fn(f64, f64) -> f64 + Clone + Send + Sync,
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
                "The `alpha` must be in the range (0, 2], got {alpha}"
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

    /// Get the stability index
    pub fn get_alpha(&self) -> f64 {
        self.alpha
    }
}

impl<D, G> ContinuousProcess for GeneralizedLangevin<D, G>
where
    D: Fn(f64, f64) -> f64 + Clone + Send + Sync,
    G: Fn(f64, f64) -> f64 + Clone + Send + Sync,
{
    fn simulate_unchecked(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_generalized_langevin(
            &self.drift_func,
            &self.diffusion_func,
            self.start_position,
            self.alpha,
            duration,
            time_step,
        )
    }

    fn displacement(&self, duration: f64, time_step: f64) -> XResult<f64> {
        let num_steps = (duration / time_step).ceil() as usize;

        let noise = stable::sym_standard_rands(self.alpha, num_steps - 1)?;
        let sigma = time_step.powf(1.0 / self.alpha);

        let mut current_t = 0.0;
        let mut current_x = self.start_position;
        for xi in noise {
            current_x += self.get_drift_func()(current_x, current_t) * time_step
                + self.get_diffusion_func()(current_x, current_t) * xi * sigma;
            current_t += time_step;
        }
        let current_t = (num_steps - 1) as f64 * time_step;
        let last_step = duration - current_t;
        let last_sigma = last_step.powf(1.0 / self.alpha);
        let xi = stable::sym_standard_rand(self.alpha)?;
        current_x += self.get_drift_func()(current_x, current_t) * last_step
            + self.get_diffusion_func()(current_x, current_t) * xi * last_sigma;
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
pub fn simulate_generalized_langevin<D, G>(
    drift: &D,
    diffusion: &G,
    start_position: f64,
    alpha: f64,
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

    let noise = stable::sym_standard_rands(alpha, num_steps - 1)?;
    let sigma = time_step.powf(1.0 / alpha);

    let mut current_x = start_position;
    let mut current_t = 0.0;
    for xi in noise {
        current_x +=
            drift(current_x, current_t) * time_step + diffusion(current_x, current_t) * xi * sigma;
        x.push(current_x);
        current_t += time_step;
        t.push(current_t);
    }

    let last_step = duration - current_t;
    let sigma = last_step.powf(1.0 / alpha);
    let xi = stable::sym_standard_rand(alpha)?;
    current_x +=
        drift(current_x, current_t) * last_step + diffusion(current_x, current_t) * xi * sigma;
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
pub struct SubordinatedLangevin<D, G>
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
    /// The stability index
    alpha: f64,
}

impl<D, G> SubordinatedLangevin<D, G>
where
    D: Fn(f64, f64) -> f64 + Clone + Send + Sync,
    G: Fn(f64, f64) -> f64 + Clone + Send + Sync,
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
                "The `alpha` must be in the range (0, 1), got {alpha}"
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

    /// Get the stability index
    pub fn get_alpha(&self) -> f64 {
        self.alpha
    }
}

impl<D, G> ContinuousProcess for SubordinatedLangevin<D, G>
where
    D: Fn(f64, f64) -> f64 + Clone + Send + Sync,
    G: Fn(f64, f64) -> f64 + Clone + Send + Sync,
{
    fn simulate_unchecked(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_subordinated_langevin(
            &self.drift_func,
            &self.diffusion_func,
            self.start_position,
            self.alpha,
            duration,
            time_step,
        )
    }

    fn displacement(&self, duration: f64, time_step: f64) -> XResult<f64> {
        let (t, s) = Subordinator::new(self.alpha)?.simulate(duration, time_step)?;
        let num_steps = t.len() - 1;
        let noise = normal::standard_rands::<f64>(num_steps);

        let mut current_x = self.start_position;
        for i in 0..num_steps {
            let ti = unsafe { *t.get_unchecked(i) };
            let xi = unsafe { *noise.get_unchecked(i) };
            let si = unsafe { *s.get_unchecked(i) };
            let si_next = unsafe { *s.get_unchecked(i + 1) };
            let delta_s = si_next - si;

            current_x += self.get_drift_func()(current_x, ti) * delta_s
                + self.get_diffusion_func()(current_x, ti) * xi * delta_s.sqrt();
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
pub fn simulate_subordinated_langevin<D, G>(
    drift: &D,
    diffusion: &G,
    start_position: f64,
    alpha: f64,
    duration: f64,
    time_step: f64,
) -> XResult<Pair>
where
    D: Fn(f64, f64) -> f64 + Send + Sync,
    G: Fn(f64, f64) -> f64 + Send + Sync,
{
    let (t, s) = Subordinator::new(alpha)?.simulate(duration, time_step)?;
    let num_steps = t.len() - 1;
    let noise = normal::standard_rands::<f64>(num_steps);

    let mut x = Vec::with_capacity(t.len());
    x.push(start_position);

    unsafe {
        let mut current_x = start_position;
        for i in 0..num_steps {
            let ti = *t.get_unchecked(i);
            let xi = *noise.get_unchecked(i);
            let si = *s.get_unchecked(i);
            let si_next = *s.get_unchecked(i + 1);
            let delta_s = si_next - si;

            current_x +=
                drift(current_x, ti) * delta_s + diffusion(current_x, ti) * xi * delta_s.sqrt();

            x.push(current_x);
        }
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
