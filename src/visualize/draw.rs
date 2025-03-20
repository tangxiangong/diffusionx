use crate::{
    PlotterError, XResult,
    simulation::prelude::*,
    visualize::{PlotConfig, PlotterBackend},
};
use plotters::prelude::*;
use std::path::Path;

/// Trait for visualizing a trajectory.
pub trait Visualize {
    /// Plot the trajectory.
    ///
    /// # Arguments
    ///
    /// * `config`: The configuration for the plot.
    fn plot(&self, config: &PlotConfig) -> XResult<()>;
}

impl<CP: ContinuousProcess> Visualize for ContinuousTrajectory<CP> {
    fn plot(&self, config: &PlotConfig) -> XResult<()> {
        ensure_output_dir(&config.output_path)?;
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

impl<P: PointProcess> Visualize for PointTrajectory<P> {
    fn plot(&self, config: &PlotConfig) -> XResult<()> {
        ensure_output_dir(&config.output_path)?;
        match config.backend {
            PlotterBackend::BitMap => {
                let backend = BitMapBackend::new(&config.output_path, config.size);
                config.stair(backend, self)
            }
            PlotterBackend::SVG => {
                let backend = SVGBackend::new(&config.output_path, config.size);
                config.stair(backend, self)
            }
        }
    }
}

/// Ensure the output directory exists, or create it if it doesn't exist.
fn ensure_output_dir(path: &Path) -> XResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| PlotterError::ConfigError(e.to_string()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        simulation::{OrnsteinUhlenbeck, Poisson},
        visualize::PlotConfigBuilder,
    };

    #[test]
    fn test_stair() {
        let duration = 10.0;
        let process = Poisson::new(1.0).unwrap().duration(duration).unwrap();
        let config = PlotConfigBuilder::default()
            .backend(PlotterBackend::SVG)
            .output_path("tmp/poisson.svg")
            .caption("Poisson")
            .show_grid(false)
            .title("泊松过程")
            .build()
            .unwrap();
        process.plot(&config).unwrap();
    }

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
            .title("中文")
            .title_font_size(40)
            .title_font_style(FontStyle::Bold)
            .build()
            .unwrap();
        ou.plot(&config).unwrap()
    }
}
