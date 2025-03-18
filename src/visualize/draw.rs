use crate::{
    XResult,
    simulation::prelude::*,
    visualize::{PlotConfig, PlotterBackend},
};
use plotters::prelude::*;

pub trait Visualize {
    fn plot(&self, config: &PlotConfig) -> XResult<()>;
}

impl<CP: ContinuousProcess> Visualize for ContinuousTrajectory<CP> {
    fn plot(&self, config: &PlotConfig) -> XResult<()> {
        match config.backend {
            PlotterBackend::BitMap => {
                let backend = BitMapBackend::new(&config.output_path, config.size);
                config.plot(backend, self)
            }
            PlotterBackend::SVG => {
                let backend = SVGBackend::new(&config.output_path, config.size);
                config.plot(backend, self)
            }
        }
    }
}

// impl<P: PointProcess> Visualize for PointTrajectory<P> {
//     fn plot(&self, config: &PlotConfig) -> XResult<()> {
//         match config.backend {
//             PlotterBackend::BitMap => {
//                 let backend = BitMapBackend::new(&config.output_path, config.size);
//                 config.stairs(backend, self)
//             }
//             PlotterBackend::SVG => {
//                 let backend = SVGBackend::new(&config.output_path, config.size);
//                 config.stairs(backend, self)
//             }
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{simulation::Bm, visualize::PlotConfigBuilder};

    #[test]
    fn test_plot() {
        let bm = Bm::default().duration(10).unwrap();
        let config = PlotConfigBuilder::default()
            .time_step(0.01)
            .output_path("bm.png")
            .backend(PlotterBackend::BitMap)
            .build()
            .unwrap();
        bm.plot(&config).unwrap()
    }
}
