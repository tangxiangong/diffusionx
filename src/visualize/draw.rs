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

impl<CP: ContinuousProcess + Clone> Visualize for ContinuousTrajectory<CP> {
    /// Plot the continuous trajectory.
    ///
    /// # Arguments
    ///
    /// * `config`: The configuration for the plot.
    fn plot(&self, config: &PlotConfig) -> XResult<()> {
        ensure_output_dir(&config.output_path)?;
        let traj = self.simulate(config.time_step)?;
        match config.backend {
            PlotterBackend::BitMap => {
                let backend = BitMapBackend::new(&config.output_path, config.size);
                config.plot(backend, traj)
            }
            PlotterBackend::SVG => {
                let backend = SVGBackend::new(&config.output_path, config.size);
                config.plot(backend, traj)
            }
        }
    }
}

impl<P: PointProcess> Visualize for PointTrajectory<P> {
    /// Plot the point trajectory.
    fn plot(&self, config: &PlotConfig) -> XResult<()> {
        ensure_output_dir(&config.output_path)?;
        let traj = self.simulate_with_duration()?;
        match config.backend {
            PlotterBackend::BitMap => {
                let backend = BitMapBackend::new(&config.output_path, config.size);
                if config.stairs {
                    config.stair(backend, traj)
                } else {
                    config.plot(backend, traj)
                }
            }
            PlotterBackend::SVG => {
                let backend = SVGBackend::new(&config.output_path, config.size);
                if config.stairs {
                    config.stair(backend, traj)
                } else {
                    config.plot(backend, traj)
                }
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
            set_config(config, backend, points, meta, false)
        }
        PlotterBackend::SVG => {
            let backend = SVGBackend::new(&config.output_path, config.size);
            set_config(config, backend, points, meta, false)
        }
    }
}

/// Plot a loglog.
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
/// let times = vec![1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0];
/// let positions = vec![2.0, 20.0, 200.0, 2000.0, 20000.0, 200000.0];
/// let config = PlotConfig::default();
/// loglog(&times, &positions, &config).unwrap();
/// ```
pub fn loglog(times: &[f64], positions: &[f64], config: &PlotConfig) -> XResult<()> {
    let max_time = *times.last().unwrap();
    let (min_x, max_x) = minmax(positions);
    let meta = (max_time, min_x, max_x);
    let points: Vec<(f64, f64)> = times.iter().zip(positions).map(|(&t, &x)| (t, x)).collect();
    match config.backend {
        PlotterBackend::BitMap => {
            let backend = BitMapBackend::new(&config.output_path, config.size);
            set_config(config, backend, points, meta, true)
        }
        PlotterBackend::SVG => {
            let backend = SVGBackend::new(&config.output_path, config.size);
            set_config(config, backend, points, meta, true)
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
            set_config(config, backend, points, meta, false)
        }
        PlotterBackend::SVG => {
            let backend = SVGBackend::new(&config.output_path, config.size);
            set_config(config, backend, points, meta, false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        simulation::{continuous::OrnsteinUhlenbeck, point::Poisson},
        visualize::PlotConfigBuilder,
    };

    #[test]
    #[ignore]
    fn test_stair() {
        let duration = 10.0;
        let process = Poisson::new(1.0).unwrap();
        let traj = process.duration(duration).unwrap();
        let config = PlotConfigBuilder::default()
            .backend(PlotterBackend::SVG)
            .output_path("tmp/poisson.svg")
            .caption("Poisson")
            .show_grid(false)
            .title("Poisson")
            .build()
            .unwrap();
        traj.plot(&config).unwrap();
    }

    #[test]
    #[ignore]
    fn test_plot() {
        let duration = 100.0;
        let ou = OrnsteinUhlenbeck::new(1.0, 1.0, 0.0).unwrap();
        let traj = ou.duration(duration).unwrap();
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
        traj.plot(&config).unwrap()
    }

    #[test]
    #[ignore]
    fn test_loglog() {
        let times = vec![1.0, 10.0, 100.0, 1000.0];
        let positions = vec![2.0, 20.0, 200.0, 2000.0];
        let config = PlotConfigBuilder::default()
            .time_step(0.01)
            .backend(PlotterBackend::SVG)
            .output_path("tmp/loglog.svg")
            .caption("loglog")
            .show_grid(false)
            .title("中文")
            .title_font_size(40)
            .title_font_style(FontStyle::Bold)
            .build()
            .unwrap();
        loglog(&times, &positions, &config).unwrap()
    }
}
