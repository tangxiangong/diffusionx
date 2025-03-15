use crate::{XResult, simulation::prelude::*, visualize::PlotConfig};

pub trait Visualize {
    fn plot(&self, config: &PlotConfig) -> XResult<()>;
}

impl<CP: ContinuousProcess> Visualize for ContinuousTrajectory<CP> {
    fn plot(&self, config: &PlotConfig) -> XResult<()> {
        let (_times, _positions) = self.simulate(config.time_step)?;
        todo!()
    }
}
