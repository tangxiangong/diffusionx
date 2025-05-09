use crate::{
    XResult,
    simulation::prelude::*,
    utils::{ensure_output_dir, minmax},
    visualize::{PlotConfig, PlotterBackend},
};
use plotters::prelude::*;

use super::set_config;

/// Trait for visualizing a trajectory.
pub trait Visualize {
    /// Plot the trajectory.
    ///
    /// # Arguments
    ///
    /// * `config`: The configuration for the plot.
    fn plot(&self, config: &PlotConfig) -> XResult<()>;
}

/// Implement the `Visualize` trait for `ContinuousTrajectory`.
impl<CP: ContinuousProcess> Visualize for ContinuousTrajectory<CP> {
    /// Plot the continuous trajectory.
    ///
    /// # Arguments
    ///
    /// * `config`: The configuration for the plot.
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

/// Implement the `Visualize` trait for `PointTrajectory`.
impl<P: PointProcess> Visualize for PointTrajectory<P> {
    /// Plot the point trajectory.
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

/// Plot a continuous trajectory.
///
/// # Arguments
///
/// * `times` - The times of the trajectory.
/// * `positions` - The positions of the trajectory.
/// * `config` - The configuration for the plot.
///
/// # Examples
///
/// ```rust
/// let times = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
/// let positions = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
/// let config = PlotConfig::default();
/// plot(&times, &positions, &config).unwrap();
/// ```
pub fn plot(times: &[f64], positions: &[f64], config: &PlotConfig) -> XResult<()> {
    let max_time = *times.last().unwrap();
    let (min_x, max_x) = minmax(positions);
    let meta = (max_time, min_x, max_x);
    let points: Vec<(f64, f64)> = times.iter().zip(positions).map(|(&t, &x)| (t, x)).collect();
    match config.backend {
        PlotterBackend::BitMap => {
            let backend = BitMapBackend::new(&config.output_path, config.size);
            set_config(config, backend, points, meta)
        }
        PlotterBackend::SVG => {
            let backend = SVGBackend::new(&config.output_path, config.size);
            set_config(config, backend, points, meta)
        }
    }
}

/// Plot a point trajectory.
///
/// # Arguments
///
/// * `times` - The times of the trajectory.
/// * `positions` - The positions of the trajectory.
/// * `config` - The configuration for the plot.
///
/// # Examples
///
/// ```rust
/// let times = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
/// let positions = vec![0, 1, 2, 3, 4, 5];
/// let config = PlotConfig::default();
/// stair(&times, &positions, &config).unwrap();
/// ```
pub fn stair(times: &[f64], positions: &[i64], config: &PlotConfig) -> XResult<()> {
    let max_time = *times.last().unwrap();
    let min_x = *positions.iter().min().unwrap() as f64;
    let max_x = *positions.iter().max().unwrap() as f64;
    let meta = (max_time, min_x, max_x);
    let points: Vec<(f64, f64)> = times
        .iter()
        .zip(positions)
        .enumerate()
        .flat_map(|(i, (&t, &y))| {
            if i == times.len() - 1 {
                vec![(t, y as f64)]
            } else {
                vec![(t, y as f64), (times[i + 1], y as f64)]
            }
        })
        .collect();
    match config.backend {
        PlotterBackend::BitMap => {
            let backend = BitMapBackend::new(&config.output_path, config.size);
            set_config(config, backend, points, meta)
        }
        PlotterBackend::SVG => {
            let backend = SVGBackend::new(&config.output_path, config.size);
            set_config(config, backend, points, meta)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        simulation::{continuous::OrnsteinUhlenbeck, jump::Poisson},
        visualize::PlotConfigBuilder,
    };

    #[test]
    #[ignore]
    fn test_stair() {
        let duration = 10.0;
        let process = Poisson::new(1.0).unwrap().duration(duration).unwrap();
        let config = PlotConfigBuilder::default()
            .backend(PlotterBackend::SVG)
            .output_path("tmp/poisson.svg")
            .caption("Poisson")
            .show_grid(false)
            .title("Poisson")
            .build()
            .unwrap();
        process.plot(&config).unwrap();
    }

    #[test]
    #[ignore]
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
