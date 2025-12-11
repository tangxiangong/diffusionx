use crate::{
    FloatExt, SimulationError, XResult, check_duration_time_step,
    simulation::prelude::ContinuousProcess,
};

/// Inverse process of a continuous process
///
/// The inverse process for a continuous monotonic process $X(t)$ is defined as the process that
///
/// $$ Y(t) = \inf \{ s\geqslant 0\ |\ X(s)\geqslant t \} $$
///
///
#[derive(Debug, Clone)]
pub struct InverseProcess<'a, CP: ContinuousProcess<T>, T: FloatExt = f64> {
    /// The process
    process: &'a CP,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, CP: ContinuousProcess<T>, T: FloatExt> InverseProcess<'a, CP, T> {
    /// Create a new inverse process with given process
    ///
    /// # Arguments
    ///
    /// * `process` - The continuous process
    pub fn new(process: &'a CP) -> Self {
        Self {
            process,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the process
    pub fn get_process(&self) -> &'a CP {
        self.process
    }
}

impl<'a, CP: ContinuousProcess<T>, T: FloatExt> ContinuousProcess<T> for InverseProcess<'a, CP, T> {
    fn displacement(&self, duration: T, time_step: T) -> XResult<T> {
        let mut mut_duration = duration;
        let two = T::from(2).unwrap();
        let (t, s) = loop {
            let (t, s) = self.process.simulate(mut_duration, time_step)?;
            let last = match s.last() {
                Some(x) => *x,
                None => return Err(SimulationError::Unknown.into()),
            };
            if last >= duration {
                break (t, s);
            }
            mut_duration *= two;
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

        let target_time = T::zero();
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

    fn end(&self, duration: T, time_step: T) -> XResult<T> {
        let mut mut_duration = duration;
        let two = T::from(2).unwrap();
        let (t, s) = loop {
            let (t, s) = self.process.simulate(mut_duration, time_step)?;
            let last = match s.last() {
                Some(x) => *x,
                None => return Err(SimulationError::Unknown.into()),
            };
            if last >= duration {
                break (t, s);
            }
            mut_duration *= two;
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
        Ok(t[pos])
    }

    fn start(&self) -> T {
        let duration = T::from(1).unwrap();
        let time_step = T::from(0.01).unwrap();
        let mut mut_duration = T::from(1).unwrap();
        let two = T::from(2).unwrap();
        let (t, s) = loop {
            let (t, s) = self.process.simulate(mut_duration, time_step).unwrap();
            let last = *s.last().unwrap();
            if last >= duration {
                break (t, s);
            }
            mut_duration *= two;
        };

        let target_time = T::zero();
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
        t[pos]
    }

    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        check_duration_time_step(duration, time_step)?;
        let two = T::from(2).unwrap();
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
            mut_duration *= two;
        };

        let num_steps = (duration / time_step).ceil().to_usize().unwrap();
        let mut inv_times: Vec<_> = (0..=num_steps)
            .map(|i| T::from(i).unwrap() * time_step)
            .collect();
        *inv_times.last_mut().unwrap() = duration;

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
pub trait Inverse<T: FloatExt>: ContinuousProcess<T> {
    /// Create a new `InverseProcess`
    fn inverse(&self) -> InverseProcess<'_, Self, T>
    where
        Self: Sized,
    {
        InverseProcess::new(self)
    }
}
