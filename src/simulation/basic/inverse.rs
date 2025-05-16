use crate::{SimulationError, XResult};

use super::{ContinuousProcess, Pair};

/// Inverse process of a continuous process
#[derive(Clone)]
pub struct InverseProcess<T: ContinuousProcess> {
    /// The process
    process: T,
}

impl<T: ContinuousProcess> InverseProcess<T> {
    /// Create a new inverse process with given process
    ///
    /// # Arguments
    ///
    /// * `process` - The continuous process
    pub fn new(process: &T) -> Self {
        Self {
            process: process.clone(),
        }
    }

    /// Get the process
    pub fn process(&self) -> &T {
        &self.process
    }
}

impl<T: ContinuousProcess> ContinuousProcess for InverseProcess<T> {
    fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<Pair> {
        let mut mut_duration = duration.into();
        let duration = mut_duration;
        let sp = self.process.clone();
        let (t, s) = loop {
            let (t, s) = sp.simulate(mut_duration, time_step)?;
            let last = match s.last() {
                Some(x) => *x,
                None => return Err(SimulationError::Unknown.into()),
            };
            if last >= duration {
                break (t, s);
            }
            mut_duration *= 2.0;
        };

        let num_steps = (duration / time_step).ceil() as usize;
        let inv_times: Vec<f64> = (0..=num_steps)
            .map(|i| {
                if i == num_steps {
                    duration
                } else {
                    i as f64 * time_step
                }
            })
            .collect();

        let mut inv_path = Vec::with_capacity(inv_times.len());

        for &target_time in &inv_times {
            let pos = match s.binary_search_by(|&x| x.partial_cmp(&target_time).unwrap()) {
                Ok(idx) => idx,
                Err(idx) => {
                    if idx >= s.len() {
                        s.len() - 1
                    } else {
                        idx
                    }
                }
            };

            inv_path.push(t[pos]);
        }

        Ok((inv_times, inv_path))
    }
}

/// The inverse process trait
pub trait Inverse: ContinuousProcess {
    /// Create a new `InverseProcess`
    fn inverse(&self) -> InverseProcess<Self> {
        InverseProcess::new(self)
    }
}
