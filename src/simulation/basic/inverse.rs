use crate::{SimulationError, XResult};

use super::{ContinuousProcess, Pair};

/// Inverse process of a continuous process
#[derive(Debug, Clone)]
pub struct InverseProcess<'a, T: ContinuousProcess> {
    /// The process
    process: &'a T,
}

impl<'a, T: ContinuousProcess> InverseProcess<'a, T> {
    /// Create a new inverse process with given process
    ///
    /// # Arguments
    ///
    /// * `process` - The continuous process
    pub fn new(process: &'a T) -> Self {
        Self { process }
    }

    /// Get the process
    pub fn get_process(&self) -> &'a T {
        self.process
    }
}

impl<'a, T: ContinuousProcess> ContinuousProcess for InverseProcess<'a, T> {
    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        let mut mut_duration = duration;
        let (t, s) = loop {
            let (t, s) = self.process.simulate(mut_duration, time_step)?;
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
    fn inverse(&self) -> InverseProcess<'_, Self>
    where
        Self: Sized,
    {
        InverseProcess::new(self)
    }
}
