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
    use crate::{simulation::OrnsteinUhlenbeck, visualize::PlotConfigBuilder};

    #[test]
    fn test_plot() {
        let duration = 100.0;
        let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0)
            .unwrap()
            .duration(duration)
            .unwrap();
        let config = PlotConfigBuilder::default()
            .time_step(0.01)
            .backend(PlotterBackend::SVG)
            .output_path("tmp/ou.svg")
            .caption("OU")
            .show_grid(false)
            .title("Ornstein-Uhlenbeck Process")
            .build()
            .unwrap();
        ou.plot(&config).unwrap()
    }
}
