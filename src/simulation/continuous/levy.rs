//! Lévy process simulation
//!
//! The Lévy process is a process with independent and stationary increments.

use crate::{
    FloatExt, SimulationError, XResult, check_duration_time_step,
    random::{
        normal,
        stable::{
            StableConstants, sample_standard_alpha, sample_standard_alpha_one,
            sample_standard_alpha_with_constants, sample_sym_standard_alpha_one,
            sample_sym_standard_alpha_with_constants,
            sample_sym_standard_alpha_with_stable_constants,
        },
    },
    simulation::prelude::*,
};
use rand::prelude::*;
use rand_distr::{Exp1, StandardNormal, uniform::SampleUniform};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Asymmetric Lévy process
#[derive(Debug, Clone)]
pub struct AsymmetricLevy<T: FloatExt = f64> {
    /// The starting position
    start_position: T,
    /// The stability index
    alpha: T,
    /// The asymmetry parameter
    beta: T,
}

impl<T: FloatExt> AsymmetricLevy<T> {
    /// Create a new `AsymmetricLevy`
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position.
    /// * `alpha` - The stability index.
    /// * `beta` - The asymmetry parameter.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::AsymmetricLevy;
    ///
    /// let levy = AsymmetricLevy::new(0.0, 1.5, 0.4).unwrap();
    /// ```
    pub fn new(start_position: T, alpha: T, beta: T) -> XResult<Self> {
        if alpha <= T::zero() || alpha > T::from(2.0).unwrap() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `alpha` must be in the range (0, 2], got {alpha:?}"
            ))
            .into());
        }
        if !(-T::one()..=T::one()).contains(&beta) {
            return Err(SimulationError::InvalidParameters(format!(
                "The `beta` must be in the range [-1, 1], got {beta:?}"
            ))
            .into());
        }
        Ok(Self {
            start_position,
            alpha,
            beta,
        })
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> T {
        self.start_position
    }

    /// Get the stability index
    pub fn get_alpha(&self) -> T {
        self.alpha
    }

    /// Get the asymmetry parameter
    pub fn get_beta(&self) -> T {
        self.beta
    }
}

impl<T: FloatExt + SampleUniform> ContinuousProcess<T> for AsymmetricLevy<T>
where
    Exp1: Distribution<T>,
    StandardNormal: Distribution<T>,
{
    fn start(&self) -> T {
        self.start_position
    }

    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        simulate_asymmetric_levy(
            self.start_position,
            self.alpha,
            self.beta,
            duration,
            time_step,
        )
    }

    fn displacement(&self, duration: T, time_step: T) -> XResult<T> {
        check_duration_time_step(duration, time_step)?;

        let power = T::one() / self.alpha;
        let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
        let sample = if (self.alpha - T::from(2).unwrap()).abs() < T::epsilon() {
            T::from(2).unwrap().sqrt() * normal::standard_rand_with_rng::<T, _>(&mut rng)
        } else if (self.alpha - T::one()).abs() < T::epsilon() {
            sample_standard_alpha_one(self.alpha, self.beta, &mut rng)
        } else {
            sample_standard_alpha(self.alpha, self.beta, &mut rng)
        };
        Ok(sample * duration.powf(power))
    }
}

/// Simulate the asymmetric Lévy process
///
/// # Arguments
///
/// * `start_position` - The starting position.
/// * `alpha` - The stability index.
/// * `beta` - The asymmetry parameter.
/// * `duration` - The duration.
/// * `time_step` - The time step.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::{continuous::AsymmetricLevy, prelude::*};
///
/// let levy = AsymmetricLevy::new(0.0, 1.5, 0.4).unwrap();
/// let (t, x) = levy.simulate(10.0, 0.1).unwrap();
/// ```
pub fn simulate_asymmetric_levy<T: FloatExt + SampleUniform>(
    start_position: T,
    alpha: T,
    beta: T,
    duration: T,
    time_step: T,
) -> XResult<(Vec<T>, Vec<T>)>
where
    Exp1: Distribution<T>,
    StandardNormal: Distribution<T>,
{
    check_duration_time_step(duration, time_step)?;

    let num_steps = (duration / time_step).ceil().to_usize().unwrap();
    let power = T::one() / alpha;
    let mut scale = time_step.powf(power);

    let mut t = Vec::with_capacity(num_steps + 1);
    let mut x = Vec::with_capacity(num_steps + 1);

    t.push(T::zero());
    x.push(start_position);

    let mut current_x = start_position;
    let mut current_t = T::zero();
    let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
    let constants = StableConstants::new(alpha, beta);
    for _ in 0..num_steps - 1 {
        let xi = if (alpha - T::from(2).unwrap()).abs() < T::epsilon() {
            T::from(2).unwrap().sqrt() * normal::standard_rand_with_rng::<T, _>(&mut rng)
        } else if (alpha - T::one()).abs() < T::epsilon() {
            sample_standard_alpha_one(alpha, beta, &mut rng)
        } else {
            sample_standard_alpha_with_constants(&constants, alpha, &mut rng)
        };
        current_x += xi * scale;
        x.push(current_x);
        current_t += time_step;
        t.push(current_t);
    }
    let last_step = duration - current_t;
    scale = last_step.powf(power);
    let xi = if (alpha - T::from(2).unwrap()).abs() < T::epsilon() {
        T::from(2).unwrap().sqrt() * normal::standard_rand_with_rng::<T, _>(&mut rng)
    } else if (alpha - T::one()).abs() < T::epsilon() {
        sample_standard_alpha_one(alpha, beta, &mut rng)
    } else {
        sample_standard_alpha_with_constants(&constants, alpha, &mut rng)
    };
    current_x += xi * scale;
    x.push(current_x);
    t.push(duration);

    Ok((t, x))
}

/// Lévy process
#[derive(Debug, Clone)]
pub struct Levy<T: FloatExt = f64> {
    /// The starting position
    start_position: T,
    /// The stability index
    alpha: T,
}

impl<T: FloatExt> Levy<T> {
    /// Create a new `Levy`
    ///
    /// # Arguments
    ///
    /// * `start_position` - The starting position.
    /// * `alpha` - The stability index.
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::continuous::Levy;
    ///
    /// let levy = Levy::new(0.0, 1.5).unwrap();
    /// ```
    pub fn new(start_position: T, alpha: T) -> XResult<Self> {
        if alpha <= T::zero() || alpha > T::from(2.0).unwrap() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `alpha` must be in the range (0, 2], got {alpha:?}"
            ))
            .into());
        }
        Ok(Self {
            start_position,
            alpha,
        })
    }

    /// Get the starting position
    pub fn get_start_position(&self) -> T {
        self.start_position
    }

    /// Get the stability index
    pub fn get_alpha(&self) -> T {
        self.alpha
    }
}

impl<T: FloatExt + SampleUniform> ContinuousProcess<T> for Levy<T>
where
    Exp1: Distribution<T>,
    StandardNormal: Distribution<T>,
{
    fn start(&self) -> T {
        self.start_position
    }

    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        simulate_levy(self.start_position, self.alpha, duration, time_step)
    }

    fn displacement(&self, duration: T, time_step: T) -> XResult<T> {
        check_duration_time_step(duration, time_step)?;

        let power = T::one() / self.alpha;
        let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
        Ok(sample_symmetric_standard(self.alpha, &mut rng) * duration.powf(power))
    }
}

#[inline]
fn sample_symmetric_standard<T, R>(alpha: T, rng: &mut R) -> T
where
    T: FloatExt + SampleUniform,
    R: Rng + ?Sized,
    Exp1: Distribution<T>,
    StandardNormal: Distribution<T>,
{
    if (alpha - T::from(2).unwrap()).abs() < T::epsilon() {
        T::from(2).unwrap().sqrt() * normal::standard_rand_with_rng::<T, _>(rng)
    } else if (alpha - T::one()).abs() < T::epsilon() {
        sample_sym_standard_alpha_one(rng)
    } else {
        let inv_alpha = T::one() / alpha;
        let one_minus_alpha_div_alpha = (T::one() - alpha) * inv_alpha;
        sample_sym_standard_alpha_with_constants(inv_alpha, one_minus_alpha_div_alpha, alpha, rng)
    }
}

/// Simulate the Lévy process
///
/// # Arguments
///
/// * `start_position` - The starting position.
/// * `alpha` - The stability index.
/// * `duration` - The duration.
/// * `time_step` - The time step.
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::continuous::levy::simulate_levy;
///
/// let (t, x) = simulate_levy(0.0, 1.5, 1.0, 0.1).unwrap();
/// ```
pub fn simulate_levy<T: FloatExt + SampleUniform>(
    start_position: T,
    alpha: T,
    duration: T,
    time_step: T,
) -> XResult<(Vec<T>, Vec<T>)>
where
    Exp1: Distribution<T>,
    StandardNormal: Distribution<T>,
{
    check_duration_time_step(duration, time_step)?;

    let num_steps = (duration / time_step).ceil().to_usize().unwrap();
    let power = T::one() / alpha;
    let mut scale = time_step.powf(power);

    let mut t = Vec::with_capacity(num_steps + 1);
    let mut x = Vec::with_capacity(num_steps + 1);
    t.push(T::zero());
    x.push(start_position);

    let mut current_x = start_position;
    let mut current_t = T::zero();
    let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
    let sym_constants = StableConstants::new(alpha, T::zero());
    for _ in 0..num_steps - 1 {
        let xi = if (alpha - T::from(2).unwrap()).abs() < T::epsilon() {
            T::from(2).unwrap().sqrt() * normal::standard_rand_with_rng::<T, _>(&mut rng)
        } else if (alpha - T::one()).abs() < T::epsilon() {
            sample_sym_standard_alpha_one(&mut rng)
        } else {
            sample_sym_standard_alpha_with_stable_constants(&sym_constants, alpha, &mut rng)
        };
        current_x += xi * scale;
        x.push(current_x);
        current_t += time_step;
        t.push(current_t);
    }
    let last_step = duration - current_t;
    scale = last_step.powf(power);
    let xi = if (alpha - T::from(2).unwrap()).abs() < T::epsilon() {
        T::from(2).unwrap().sqrt() * normal::standard_rand_with_rng::<T, _>(&mut rng)
    } else if (alpha - T::one()).abs() < T::epsilon() {
        sample_sym_standard_alpha_one(&mut rng)
    } else {
        sample_sym_standard_alpha_with_stable_constants(&sym_constants, alpha, &mut rng)
    };
    current_x += xi * scale;
    x.push(current_x);
    t.push(duration);

    Ok((t, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_levy() {
        let levy = Levy::new(10.0, 1.5).unwrap();
        let asymmetric_levy = AsymmetricLevy::new(10.0, 1.5, 0.4).unwrap();
        let time_step = 0.1;
        let duration = 1.0;
        let (t, x) = levy.simulate(duration, time_step).unwrap();
        println!("t: {t:?}");
        println!("x: {x:?}");
        let (t, x) = asymmetric_levy.simulate(duration, time_step).unwrap();
        println!("t: {t:?}");
        println!("x: {x:?}");
    }

    #[test]
    fn test_fpt() {
        let levy = Levy::new(0.0, 1.5).unwrap();
        let asymmetric_levy = AsymmetricLevy::new(0.0, 1.5, 0.4).unwrap();
        let time_step = 0.1;
        let fpt = levy.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {fpt:?}");
        let fpt = asymmetric_levy.fpt((-1.0, 1.0), 1000.0, time_step).unwrap();
        println!("fpt: {fpt:?}");
    }

    #[test]
    fn test_occupation_time() {
        let levy = Levy::new(0.0, 1.5).unwrap();
        let asymmetric_levy = AsymmetricLevy::new(0.0, 1.5, 0.4).unwrap();
        let time_step = 0.1;
        let ot = levy.occupation_time((-1.0, 1.0), 10.0, time_step).unwrap();
        println!("ot: {ot:?}");
        let ot = asymmetric_levy
            .occupation_time((-1.0, 1.0), 10.0, time_step)
            .unwrap();
        println!("ot: {ot:?}");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Levy>();
        assert_send_sync::<AsymmetricLevy>();
    }
}
