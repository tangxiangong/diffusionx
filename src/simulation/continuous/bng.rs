//! Brownian yet non-Gaussian process simulation

use crate::{
    XResult, check_duration_time_step,
    random::normal,
    simulation::{continuous::simulate_ou, prelude::*},
};

/// Brownian yet non-Gaussian process
///
/// $$dr(t) = \sqrt{2 * D(t)} dW_1(t), \quad r(0) = r0,$$
///
/// $$D(t) = Y(t)^2,$$
///
/// $$dY(t) = -Y(t) dt + dW_2(t), \quad Y(0) = Y0,$$
///
/// where $W_1(t)$ and $W_2(t)$ are two independent Wiener processes.
#[derive(Debug, Clone)]
pub struct BnG {
    /// The starting position.
    start_position: f64,
    /// The starting position of the OU process.
    ou_start_position: f64,
}

impl BnG {
    /// Create a new `BnG`
    ///
    /// # Arguments
    ///
    /// * `start_position` - The initial position r0 of the process.
    /// * `ou_start_position` - The initial position Y0 of the OU process.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::BnG;
    ///
    /// let bng = BnG::new(0.0, 1.0).unwrap();
    /// ```
    pub fn new(start_position: impl Into<f64>, ou_start_position: impl Into<f64>) -> XResult<Self> {
        let start_position = start_position.into();
        let ou_start_position = ou_start_position.into();
        Ok(Self {
            start_position,
            ou_start_position,
        })
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> f64 {
        self.start_position
    }

    /// Get the starting position of the OU process
    pub fn get_ou_start_position(&self) -> f64 {
        self.ou_start_position
    }
}

impl ContinuousProcess for BnG {
    fn start(&self) -> f64 {
        self.start_position
    }

    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        simulate_bng(
            self.start_position,
            self.ou_start_position,
            duration,
            time_step,
        )
    }

    fn displacement(&self, duration: f64, time_step: f64) -> XResult<f64> {
        let (_, y) = simulate_ou(1.0, 1.0, self.ou_start_position, duration, time_step)?;
        let noise = normal::standard_rands::<f64>(y.len() - 1);
        Ok(y.into_iter()
            .skip(1)
            .zip(noise)
            .fold(0.0, |state, (yi, xi)| {
                state + yi.abs() * (2.0 * time_step).sqrt() * xi
            }))
    }
}

/// Simulate the Brownian yet non-Gaussian process
///
/// # Mathematical Formulation
///
/// $$dr(t) = \sqrt{2 * D(t)} dW_1(t), \quad r(0) = r0,$$
///
/// $$D(t) = Y(t)^2,$$
///
/// $$dY(t) = -Y(t) dt + dW_2(t), \quad Y(0) = Y0,$$
///
/// where $W_1(t)$ and $W_2(t)$ are two independent Wiener processes.
///
/// # Arguments
///
/// * `start_position` - The starting position.
/// * `ou_start_position` - The starting position of the OU process.
/// * `duration` - The duration.
/// * `time_step` - The time step.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::bng::simulate_bng;
///
/// let (t, x) = simulate_bng(0.0, 1.0, 1.0, 0.01).unwrap();
/// ```
pub fn simulate_bng(
    start_position: f64,
    ou_start_position: f64,
    duration: f64,
    time_step: f64,
) -> XResult<Pair> {
    check_duration_time_step(duration, time_step)?;

    let num_steps = (duration / time_step).ceil() as usize;

    let mut t = Vec::with_capacity(num_steps + 1);
    let mut x = Vec::with_capacity(num_steps + 1);

    t.push(0.0);
    x.push(start_position);

    let mut scale_ou = time_step.sqrt();
    let mut scale_bng = (2.0 * time_step).sqrt();

    let mut current_t = 0.0;
    let mut current_x = start_position;
    let mut current_y = ou_start_position;

    let noises_ou = normal::standard_rands::<f64>(num_steps - 1);
    let noises_bng = normal::standard_rands::<f64>(num_steps - 1);

    for (xi_ou, xi_bng) in noises_ou.into_iter().zip(noises_bng) {
        current_y += -current_x * time_step + xi_ou * scale_ou;
        current_t += time_step;
        current_x += current_y.abs() * scale_bng * xi_bng;
        t.push(current_t);
        x.push(current_x);
    }

    let last_step = duration - current_t;
    scale_ou = last_step.sqrt();
    scale_bng = (2.0 * last_step).sqrt();
    current_y += -current_x * last_step + normal::standard_rand::<f64>() * scale_ou;
    current_x += current_y.abs() * scale_bng * normal::standard_rand::<f64>();
    x.push(current_x);
    t.push(duration);

    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_bng() {
        let bng = BnG::new(0.0, 1.0).unwrap();
        let (t, x) = bng.simulate(1.0, 0.01).unwrap();
        assert_eq!(t.len(), x.len());
        assert!(t.last().unwrap() <= &1.0);
    }

    #[test]
    fn test_mean() {
        let bng = BnG::new(0.0, 1.0).unwrap();
        let _mean = bng.mean(1.0, 1000, 0.01).unwrap();
    }

    #[test]
    fn test_msd() {
        let bng = BnG::new(0.0, 1.0).unwrap();
        let msd = bng.msd(1.0, 1000, 0.01).unwrap();
        assert!(msd > 0.0);
    }

    #[test]
    fn test_raw_moment() {
        let bng = BnG::new(0.0, 1.0).unwrap();
        let _raw_moment = bng.raw_moment(1.0, 1, 1000, 0.01).unwrap();
        // 由于随机过程的特性，这里不做具体数值断言
    }

    #[test]
    fn test_central_moment() {
        let bng = BnG::new(0.0, 1.0).unwrap();
        let central_moment = bng.central_moment(1.0, 2, 1000, 0.01).unwrap();
        assert!(central_moment > 0.0);
    }

    #[test]
    fn test_fpt() {
        let bng = BnG::new(0.0, 1.0).unwrap();
        let _fpt = bng.fpt((-1.0, 1.0), 10.0, 0.01).unwrap();
    }

    #[test]
    fn test_occupation_time() {
        let bng = BnG::new(0.0, 1.0).unwrap();
        let occupation_time = bng.occupation_time((-1.0, 1.0), 1.0, 0.01).unwrap();
        assert!((0.0..=1.0).contains(&occupation_time));
    }
}
