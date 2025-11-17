use crate::{
    SimulationError, XResult,
    simulation::prelude::{ContinuousProcess, Pair},
};

/// Inverse process of a continuous process
///
/// The inverse process for a continuous monotonic process $X(t)$ is defined as the process that
///
/// $$ Y(t) = \inf \{ s\geqslant 0\ |\ X(s)\geqslant t \} $$
///
///
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
    fn displacement(&self, duration: f64, time_step: f64) -> XResult<f64> {
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

        let target_time = duration;
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
        let end = t[pos];

        let target_time = 0.0;
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

        let start = t[pos];

        Ok(end - start)
    }

    fn start(&self) -> f64 {
        panic!("Not implemented")
    }

    fn end(&self, duration: f64, time_step: f64) -> XResult<f64> {
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

        let target_time = duration;
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
        let end = t[pos];

        Ok(end)
    }

    fn simulate_unchecked(&self, duration: f64, time_step: f64) -> XResult<Pair> {
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
