//! Fractional Brownian motion simulation

use crate::{
    FloatExt, SimulationError, XResult, check_duration_time_step,
    simulation::{continuous::Bm, prelude::*},
    utils::{CirculantEmbedding, cumsum},
};
use rand_distr::{Distribution, StandardNormal};
use realfft::FftNum;

/// Fractional Brownian motion
#[derive(Debug, Clone)]
pub struct FBm<T: FloatExt = f64> {
    /// The starting position
    start_position: T,
    /// The Hurst exponent
    hurst_exponent: T,
}

impl<T: FloatExt> FBm<T> {
    /// Create a new `FBm`
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position.
    /// * `hurst_exponent` - The Hurst exponent.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::Fbm;
    ///
    /// let fbm = Fbm::new(10.0, 0.5).unwrap();
    /// ```
    pub fn new(start_position: T, hurst_exponent: T) -> XResult<Self> {
        if hurst_exponent <= T::zero() || hurst_exponent >= T::one() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `hurst_exponent` must be in the range (0, 1), got {hurst_exponent:?}"
            ))
            .into());
        }
        Ok(Self {
            start_position,
            hurst_exponent,
        })
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> T {
        self.start_position
    }

    /// Get the Hurst exponent
    pub fn get_hurst_exponent(&self) -> T {
        self.hurst_exponent
    }
}

impl<T: FloatExt + FftNum> ContinuousProcess<T> for FBm<T>
where
    StandardNormal: Distribution<T>,
{
    fn start(&self) -> T {
        self.start_position
    }

    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        simulate_fbm(
            self.start_position,
            self.hurst_exponent,
            duration,
            time_step,
        )
    }

    fn displacement(&self, duration: T, time_step: T) -> XResult<T> {
        if self.hurst_exponent == T::from(0.5).unwrap() {
            let bm = Bm::default();
            return bm.displacement(duration, time_step);
        }

        let num_steps = ((duration / time_step).ceil().to_usize().unwrap()).max(1);
        let dt = duration / T::from(num_steps).unwrap();

        // Fractional Gaussian noise with covariance determined by dt and H
        let mut circulant =
            CirculantEmbedding::new(num_steps, fbm_correlation(self.hurst_exponent, dt));
        let noise = circulant.generate()?;
        let x = noise.into_iter().sum::<T>();

        Ok(x)
    }
}

/// Simulate FBM
///
/// # Arguments
///
/// * `start_position` - The starting position.
/// * `hurst_exponent` - The Hurst exponent.
/// * `duration` - The duration of the trajectory.
/// * `time_step` - The time step of the simulation.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::fbm::simulate_fbm;
///
/// let start_position = 10.0;
/// let hurst_exponent = 0.5;
/// let duration = 1.0;
/// let time_step = 0.1;
/// let (t, x) = simulate_fbm(start_position, hurst_exponent, duration, time_step).unwrap();
/// ```
pub fn simulate_fbm<T: FloatExt + FftNum>(
    start_position: T,
    hurst_exponent: T,
    duration: T,
    time_step: T,
) -> XResult<(Vec<T>, Vec<T>)>
where
    StandardNormal: Distribution<T>,
{
    check_duration_time_step(duration, time_step)?;

    if hurst_exponent == T::from(0.5).unwrap() {
        // Delegate to standard Brownian motion with D = 0.5 so Var[B(t)] = t
        return crate::simulation::continuous::bm::simulate_bm(
            start_position,
            T::from(0.5).unwrap(),
            duration,
            time_step,
        );
    }

    // Enforce a uniform time step for H != 0.5
    let num_steps = ((duration / time_step).ceil().to_usize().unwrap()).max(1);
    let dt = duration / T::from(num_steps).unwrap();

    // Uniform time grid [0, duration] with step dt
    let mut t = Vec::with_capacity(num_steps + 1);
    for i in 0..=num_steps {
        t.push(T::from(i).unwrap() * dt);
    }

    // Fractional Gaussian noise with covariance determined by dt and H
    let mut circulant = CirculantEmbedding::new(num_steps, fbm_correlation(hurst_exponent, dt));
    let noise = circulant.generate()?;

    // Calculate the cumulative sum to obtain fBm
    let x = cumsum(start_position, &noise);

    Ok((t, x))
}

/// Fractional Brownian motion correlation function
fn fbm_correlation<T: FloatExt>(hurst: T, time_step: T) -> impl Fn(usize) -> T {
    move |k: usize| {
        let two = T::from(2.0).unwrap();
        let h2 = two * hurst;
        T::from(0.5).unwrap()
            * time_step.powf(h2)
            * ((T::from(k + 1).unwrap()).powf(h2) - two * (T::from(k).unwrap()).powf(h2)
                + (T::from(k).unwrap() - T::one()).abs().powf(h2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::prelude::Moment;

    #[test]
    fn test_simulate_fbm() {
        let fbm = FBm::new(10.0, 0.5).unwrap();
        let duration = 1.0;
        let time_step = 0.1;
        let (t, x) = fbm.simulate(duration, time_step).unwrap();
        println!("t: {t:?}");
        println!("x: {x:?}");
    }

    #[test]
    fn test_raw_moment() {
        let fbm = FBm::new(10.0, 0.5).unwrap();
        let duration = 1.0;
        let time_step = 0.1;
        let traj = fbm.duration(duration).unwrap();
        let moment = traj.raw_moment(1, 1000, time_step).unwrap();
        println!("moment: {moment:?}");
    }

    #[test]
    fn test_fpt() {
        let fbm = FBm::new(0.0, 0.5).unwrap();
        let time_step = 0.1;
        let fpt = fbm.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {fpt:?}");
    }

    #[test]
    fn test_occupation_time() {
        let fbm = FBm::new(0.0, 0.5).unwrap();
        let time_step = 0.1;
        let ot = fbm.occupation_time((-1.0, 1.0), 10.0, time_step).unwrap();
        println!("ot: {ot:?}");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FBm>();
    }
}
