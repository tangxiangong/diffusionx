use crate::{SimulationError, XResult};
use num_traits::Num;
use rayon::prelude::*;

pub type Pair<Time> = (Vec<Time>, Vec<f64>);

/// Simulation trait
///
/// This trait represents a simulation.
///
/// # Arguments
///
/// * `Parameters` - The parameters of the simulation.
/// * `Results` - The results of the simulation.
///
/// # Returns
///
/// The results of the simulation.
pub trait Simulation {
    type Time: Num;
    type Params;
    fn simulate(&self, params: Self::Params) -> XResult<Pair<Self::Time>>;

    fn simulate_unchecked(&self, params: Self::Params) -> XResult<Pair<Self::Time>>;
}

pub trait CheckedParams: Simulation {
    fn check_params(&self, params: Self::Params) -> XResult<()>;
}

pub trait Moment: Simulation {
    fn raw_moment(&self, params: Self::Params, order: i32, particles: usize) -> XResult<f64>;
    fn central_moment(&self, params: Self::Params, order: i32, particles: usize) -> XResult<f64>;
}

impl<XT> Moment for XT
where
    XT: Send + Sync + Simulation + CheckedParams,
    XT::Params: Copy + Send + Sync,
{
    fn raw_moment(&self, params: Self::Params, order: i32, particles: usize) -> XResult<f64> {
        self.check_params(params)?;
        if particles == 0 {
            return Err(SimulationError::InvalidParameters(
                "particles must be positive".to_string(),
            )
            .into());
        }
        if order < 0 {
            return Err(SimulationError::InvalidParameters(
                "order must be non-negative".to_string(),
            )
            .into());
        }
        if order == 0 {
            return Ok(0.0);
        }

        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let (_, x) = self.simulate_unchecked(params)?;
                let end_position = x.last();
                match end_position {
                    Some(position) => Ok(position.powi(order)),
                    None => Err(SimulationError::Unknown.into()),
                }
            })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / particles as f64;
        Ok(result)
    }
    fn central_moment(&self, params: Self::Params, order: i32, particles: usize) -> XResult<f64> {
        let mean = self.raw_moment(params, 1, particles)?;
        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let (_, x) = self.simulate_unchecked(params)?;
                let end_position = x.last();
                match end_position {
                    Some(position) => Ok((position - mean).powi(order)),
                    None => Err(SimulationError::Unknown.into()),
                }
            })
            .try_fold(|| 0.0, |acc, res| res.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))?
            / particles as f64;
        Ok(result)
    }
}
