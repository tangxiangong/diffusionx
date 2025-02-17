use crate::{SimulationError, XResult, utils::minmax};
use rayon::prelude::*;

pub type Pair = (Vec<f64>, Vec<f64>);

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
pub trait Simulation: Clone {
    fn get_duration(&self) -> f64;
    fn mut_duration(&mut self, duration: f64);
    fn simulate(&self, time_step: f64) -> XResult<Pair>;
    fn simulate_check(&self, time_step: f64) -> XResult<Pair>;
}

pub trait CheckedParams: Simulation {
    fn check_params(&self, time_step: f64) -> XResult<()>;
}

pub trait Moment: Simulation {
    fn raw_moment(&self, time_step: f64, order: i32, particles: usize) -> XResult<f64>;
    fn central_moment(&self, time_step: f64, order: i32, particles: usize) -> XResult<f64>;
}

impl<XT> Moment for XT
where
    XT: Send + Sync + Simulation + CheckedParams,
{
    fn raw_moment(&self, time_step: f64, order: i32, particles: usize) -> XResult<f64> {
        self.check_params(time_step)?;
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
                let (_, x) = self.simulate(time_step)?;
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
    fn central_moment(&self, time_step: f64, order: i32, particles: usize) -> XResult<f64> {
        let mean = self.raw_moment(time_step, 1, particles)?;
        let result = (0..particles)
            .into_par_iter()
            .map(|_| -> XResult<f64> {
                let (_, x) = self.simulate(time_step)?;
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

pub trait Functional: Simulation + CheckedParams {
    fn fpt(&self, time_step: f64, domain: (impl Into<f64>, impl Into<f64>)) -> XResult<f64>;
    fn fpt_check(&self, time_step: f64, domain: (impl Into<f64>, impl Into<f64>)) -> XResult<f64>;
}

impl<XT: Simulation + CheckedParams> Functional for XT {
    fn fpt_check(&self, time_step: f64, domain: (impl Into<f64>, impl Into<f64>)) -> XResult<f64> {
        let (a, b) = domain;
        let a = a.into();
        let b = b.into();
        if a >= b {
            return Err(SimulationError::InvalidParameters(
                "domain must be a valid interval".to_string(),
            )
            .into());
        }
        self.check_params(time_step)?;
        self.fpt(time_step, (a, b))
    }

    fn fpt(&self, time_step: f64, domain: (impl Into<f64>, impl Into<f64>)) -> XResult<f64> {
        let (a, b) = domain;
        let a = a.into();
        let b = b.into();
        let mut duration = self.get_duration();
        let mut tmp = self.clone();
        let (t, x) = loop {
            let (t, x) = tmp.simulate(time_step)?;
            let (x_min, x_max) = minmax(&x);
            if x_min <= a || x_max >= b {
                break (t, x);
            }
            duration *= 2.0;
            tmp.mut_duration(duration);
        };
        let index = x.iter().position(|&x| x <= a || x >= b).unwrap();
        let first_passage_time = t[index];
        Ok(first_passage_time)
    }
}
