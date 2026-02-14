//! Random walk simulation

use crate::{
    RealExt, SimulationError, XResult,
    random::{PAR_THRESHOLD, exponential, stable},
    simulation::prelude::*,
    utils::cumsum,
};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Exp1, uniform::SampleUniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// Lattice random walk
///
/// # Mathematical Formulation
///
/// A lattice random walk is a stochastic process that describes a path consisting of a succession of constant steps.
/// Mathematically, it can be represented as:
///
/// $$X_n = X_0 + d_n^{(p)} \sum_{i=1}^{n} a$$
///
/// where:
/// - $X_n$ is the position after $n$ steps
/// - $X_0$ is the initial position
/// - $a$ is the step size
/// - $d_n^{(p)}$ is the direction of the step, which is either $+1$ or $-1$ with probability $p$ or $1-p$ respectively
#[derive(Clone, Debug)]
pub struct LatticeRandomWalk<T: RealExt = f64> {
    /// The step size
    step_size: T,
    /// The probability of the step in the positive direction
    probability: T,
    /// The starting position
    start_position: T,
}

impl<T: RealExt> Default for LatticeRandomWalk<T> {
    fn default() -> Self {
        Self {
            step_size: T::one(),
            probability: T::from(0.5).unwrap(),
            start_position: T::zero(),
        }
    }
}

impl<T: RealExt> LatticeRandomWalk<T> {
    /// Create a new `LatticeRandomWalk`
    ///
    /// # Arguments
    ///
    /// * `step_size` - The step size
    /// * `probability` - The probability of the step in the positive direction
    /// * `start_position` - The starting position
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::LatticeRandomWalk;
    ///
    /// let rw = LatticeRandomWalk::new(1.0, 0.5, 0.0).unwrap();
    /// ```
    pub fn new(step_size: T, probability: T, start_position: T) -> XResult<Self> {
        if probability <= T::zero() || probability > T::one() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `probability` must be between 0 and 1, got {probability:?}"
            ))
            .into());
        }
        if step_size <= T::zero() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `step_size` must be greater than 0, got {step_size:?}"
            ))
            .into());
        }
        Ok(Self {
            step_size,
            probability,
            start_position,
        })
    }

    /// Get the step size
    pub fn step_size(&self) -> T {
        self.step_size
    }

    /// Get the probability of the step in the positive direction
    pub fn probability(&self) -> T {
        self.probability
    }

    /// Get the starting position
    pub fn start_position(&self) -> T {
        self.start_position
    }
}

impl<N: IntExt, X: RealExt + std::ops::Neg<Output = X>> DiscreteProcess<N, X>
    for LatticeRandomWalk<X>
where
    std::ops::Range<N>: rayon::iter::IntoParallelIterator,
    std::ops::Range<N>: std::iter::IntoIterator,
{
    fn start(&self) -> X {
        self.start_position
    }

    fn simulate(&self, num_step: N) -> XResult<Vec<X>> {
        simulate_lattice_random_walk(
            self.step_size,
            self.probability,
            self.start_position,
            num_step,
        )
    }

    fn displacement(&self, num_step: N) -> XResult<X> {
        let prob = self.probability.to_f64().unwrap();
        let delta_x = if num_step.to_usize().unwrap() <= PAR_THRESHOLD {
            let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
            (N::zero()..num_step)
                .into_iter()
                .map(|_| rng.random_bool(prob))
                .map(|x| if x { self.step_size } else { -self.step_size })
                .sum()
        } else {
            (N::zero()..num_step)
                .into_par_iter()
                .map_init(
                    || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                    |r, _| r.random_bool(prob),
                )
                .map(|x| if x { self.step_size } else { -self.step_size })
                .sum()
        };
        Ok(delta_x)
    }
}

/// Simulate the lattice random walk
///
/// # Arguments
///
/// * `step_size` - The step size
/// * `probability` - The probability of the step in the positive direction
/// * `start_position` - The starting position
/// * `num_step` - The number of steps
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::lattice_random_walk::simulate_lattice_random_walk;
///
/// let (t, x) = simulate_lattice_random_walk(0.5, 0.5, 0.0, 1000).unwrap();
/// ```
pub fn simulate_lattice_random_walk<N: IntExt, X: RealExt + std::ops::Neg<Output = X>>(
    step_size: X,
    probability: X,
    start_position: X,
    num_step: N,
) -> XResult<Vec<X>>
where
    std::ops::Range<N>: rayon::iter::IntoParallelIterator,
    std::ops::Range<N>: std::iter::IntoIterator,
{
    let prob = probability.to_f64().unwrap();
    let delta_x: Vec<X> = if num_step.to_usize().unwrap() <= PAR_THRESHOLD {
        let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
        (N::zero()..num_step)
            .into_iter()
            .map(|_| rng.random_bool(prob))
            .map(|x| if x { step_size } else { -step_size })
            .collect()
    } else {
        (N::zero()..num_step)
            .into_par_iter()
            .map_init(
                || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                |r, _| r.random_bool(prob),
            )
            .map(|x| if x { step_size } else { -step_size })
            .collect()
    };

    let x = cumsum(start_position, &delta_x);
    Ok(x)
}

/// Random walk
///
/// # Mathematical Formulation
///
/// A stable random walk is a stochastic process that describes a path consisting of a succession of steps.
/// Mathematically, it can be represented as:
///
/// $$X_n = X_0 + \sum_{i=1}^{n} a_i$$
///
/// where:
/// - $X_n$ is the position after $n$ steps
/// - $X_0$ is the initial position
/// - $a_i$ is the step size
#[derive(Clone, Debug)]
pub struct RandomWalk<T: FloatExt = f64> {
    /// The probability of the step in the positive direction
    probability: T,
    /// The alpha parameter of the stable distribution
    alpha: T,
    /// The starting position
    start_position: T,
}

impl<T: FloatExt> Default for RandomWalk<T> {
    fn default() -> Self {
        Self {
            probability: T::from(0.5).unwrap(),
            alpha: T::from(2).unwrap(),
            start_position: T::zero(),
        }
    }
}

impl<T: FloatExt> RandomWalk<T> {
    /// Create a new `RandomWalk`
    ///
    /// # Arguments
    ///
    /// * `probability` - The probability of the step in the positive direction
    /// * `alpha` - The alpha parameter of the stable distribution
    /// * `start_position` - The starting position
    ///
    /// # Example
    ///
    /// ```rust
    /// use diffusionx::simulation::RandomWalk;
    ///
    /// let rw = RandomWalk::new(0.5, 1.0, 0.0).unwrap();
    /// ```
    pub fn new(probability: T, alpha: T, start_position: T) -> XResult<Self> {
        if probability <= T::zero() || probability > T::one() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `probability` must be between 0 and 1, got {probability:?}"
            ))
            .into());
        }
        if alpha <= T::zero() || alpha > T::from(2).unwrap() {
            return Err(SimulationError::InvalidParameters(format!(
                "The `alpha` must be between 0 and 2, got {alpha:?}"
            ))
            .into());
        }
        Ok(Self {
            probability,
            alpha,
            start_position,
        })
    }

    /// Get the probability of the step in the positive direction
    pub fn probability(&self) -> T {
        self.probability
    }

    /// Get the alpha parameter of the stable distribution
    pub fn alpha(&self) -> T {
        self.alpha
    }

    /// Get the starting position
    pub fn start_position(&self) -> T {
        self.start_position
    }
}

impl<N: IntExt, X: FloatExt + SampleUniform> DiscreteProcess<N, X> for RandomWalk<X>
where
    Exp1: Distribution<X>,
    std::ops::Range<N>: rayon::iter::IntoParallelIterator,
    std::ops::Range<N>: std::iter::IntoIterator,
{
    fn start(&self) -> X {
        self.start_position
    }

    fn simulate(&self, num_step: N) -> XResult<Vec<X>> {
        simulate_random_walk(self.probability, self.alpha, self.start_position, num_step)
    }

    fn displacement(&self, num_step: N) -> XResult<X> {
        let prob = self.probability.to_f64().unwrap();

        let delta_x = if num_step.to_usize().unwrap() <= PAR_THRESHOLD {
            let mut r = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
            if self.alpha == X::one() {
                (N::zero()..num_step)
                    .into_iter()
                    .map(|_| {
                        let dir = r.random_bool(prob);
                        if dir {
                            exponential::standard_rand().abs()
                        } else {
                            -exponential::standard_rand().abs()
                        }
                    })
                    .sum()
            } else {
                (N::zero()..num_step)
                    .into_iter()
                    .map(|_| {
                        let dir = r.random_bool(prob);
                        if dir {
                            stable::skew_rand(self.alpha).unwrap().abs()
                        } else {
                            -stable::skew_rand(self.alpha).unwrap().abs()
                        }
                    })
                    .sum()
            }
        } else if self.alpha == X::one() {
            (N::zero()..num_step)
                .into_par_iter()
                .map_init(
                    || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                    |r, _| r.random_bool(prob),
                )
                .map(|x| {
                    if x {
                        exponential::standard_rand().abs()
                    } else {
                        -exponential::standard_rand().abs()
                    }
                })
                .sum()
        } else {
            (N::zero()..num_step)
                .into_par_iter()
                .map_init(
                    || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                    |r, _| r.random_bool(prob),
                )
                .map(|x| {
                    if x {
                        stable::skew_rand(self.alpha).unwrap().abs()
                    } else {
                        -stable::skew_rand(self.alpha).unwrap().abs()
                    }
                })
                .sum()
        };
        Ok(delta_x)
    }
}

/// Simulate the random walk
///
/// # Arguments
///
/// * `probability` - The probability of the step in the positive direction
/// * `alpha` - The alpha parameter of the stable distribution
/// * `start_position` - The starting position
/// * `num_step` - The number of steps
///
/// # Example
///
/// ```rust
/// use diffusionx::simulation::random_walk::simulate_random_walk;
///
/// let (t, x) = simulate_random_walk(0.5, 1.0, 0.0, 1000).unwrap();
/// ```
pub fn simulate_random_walk<N: IntExt, X: FloatExt + SampleUniform>(
    probability: X,
    alpha: X,
    start_position: X,
    num_step: N,
) -> XResult<Vec<X>>
where
    Exp1: Distribution<X>,
    std::ops::Range<N>: rayon::iter::IntoParallelIterator,
    std::ops::Range<N>: std::iter::IntoIterator,
{
    let prob = probability.to_f64().unwrap();
    let delta_x: Vec<_> = if num_step.to_usize().unwrap() <= PAR_THRESHOLD {
        let mut r = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
        if alpha == X::one() {
            (N::zero()..num_step)
                .into_iter()
                .map(|_| {
                    let dir = r.random_bool(prob);
                    if dir {
                        exponential::standard_rand().abs()
                    } else {
                        -exponential::standard_rand().abs()
                    }
                })
                .collect()
        } else {
            (N::zero()..num_step)
                .into_iter()
                .map(|_| {
                    let dir = r.random_bool(prob);
                    if dir {
                        stable::skew_rand(alpha).unwrap().abs()
                    } else {
                        -stable::skew_rand(alpha).unwrap().abs()
                    }
                })
                .collect()
        }
    } else if alpha == X::one() {
        (N::zero()..num_step)
            .into_par_iter()
            .map_init(
                || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                |r, _| r.random_bool(prob),
            )
            .map(|x| {
                if x {
                    exponential::standard_rand().abs()
                } else {
                    -exponential::standard_rand().abs()
                }
            })
            .collect()
    } else {
        (N::zero()..num_step)
            .into_par_iter()
            .map_init(
                || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                |r, _| r.random_bool(prob),
            )
            .map(|x| {
                if x {
                    stable::skew_rand(alpha).unwrap().abs()
                } else {
                    -stable::skew_rand(alpha).unwrap().abs()
                }
            })
            .collect()
    };
    let x = cumsum(start_position, &delta_x);
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_lattice_random_walk() {
        let rw: LatticeRandomWalk<f64> = LatticeRandomWalk::default();
        let x = rw.simulate(1000).unwrap();
        assert_eq!(x.len(), 1001);
    }

    #[test]
    fn test_mean() {
        let rw: LatticeRandomWalk<f64> = LatticeRandomWalk::default();
        let _mean = rw.mean(1000, 1000).unwrap();
    }

    #[test]
    fn test_msd() {
        let rw: LatticeRandomWalk<f64> = LatticeRandomWalk::default();
        let _msd = rw.msd(1000, 1000).unwrap();
    }

    #[test]
    fn test_raw_moment() {
        let rw: LatticeRandomWalk<f64> = LatticeRandomWalk::default();
        let _moment = rw.raw_moment(1000, 1, 1000).unwrap();
    }

    #[test]
    fn test_central_moment() {
        let rw: LatticeRandomWalk<f64> = LatticeRandomWalk::default();
        let _moment = rw.central_moment(1000, 2, 1000).unwrap();
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LatticeRandomWalk>();
        assert_send_sync::<RandomWalk>();
    }
}
