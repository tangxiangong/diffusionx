use crate::{SimulationError, XResult, simulation::prelude::*, visualize::config::PlotConfig};
use plotters::prelude::*;
use plotters_backend::{DrawingBackend, DrawingErrorKind};
use std::error::Error;
use std::fmt;

/// Error type for drawing operations
#[derive(Debug)]
pub struct DrawError(Box<dyn Error + Send + Sync>);

impl fmt::Display for DrawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for DrawError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&*self.0)
    }
}

impl<E: Error + Send + Sync + 'static> From<E> for DrawError {
    fn from(err: E) -> Self {
        DrawError(Box::new(err))
    }
}

impl From<DrawError> for SimulationError {
    fn from(err: DrawError) -> Self {
        SimulationError::InvalidInput(format!("Drawing error: {}", err))
    }
}

/// Plotter trait, defines the basic functionality of plotting
pub trait Plotter {
    /// Plot the trajectory and save it to a file
    fn plot(&self, config: &PlotConfig) -> XResult<()>;

    /// Plot multiple trajectories and save them to a file
    fn plot_multi(&self, config: &PlotConfig, num_trajectories: usize) -> XResult<()>;
}

/// Continuous process plotter
pub struct ContinuousPlotter<P: ContinuousProcess> {
    /// Trajectory object
    pub trajectory: ContinuousTrajectory<P>,
    /// Time step
    pub time_step: f64,
}

impl<P: ContinuousProcess> ContinuousPlotter<P> {
    /// Create a new continuous process plotter
    pub fn new(trajectory: ContinuousTrajectory<P>, time_step: f64) -> Self {
        Self {
            trajectory,
            time_step,
        }
    }

    /// Draw a single trajectory
    fn draw_single_trajectory<DB: DrawingBackend, CT: CoordTranslate>(
        &self,
        chart: &mut ChartContext<DB, CT>,
        config: &PlotConfig,
    ) -> XResult<()>
    where
        CT::From: From<(f64, f64)>,
    {
        let (times, positions) = self.trajectory.simulate(self.time_step)?;

        let style = ShapeStyle {
            color: parse_color(&config.line_color)?,
            filled: false,
            stroke_width: config.line_width,
        };

        if config.use_step_plot {
            // 使用阶梯图绘制
            chart.draw_series(
                times
                    .iter()
                    .zip(positions.iter())
                    .map(|(&x, &y)| Circle::new((x, y), config.point_size, style))
                    .collect::<Vec<_>>(),
            )?;

            chart
                .draw_series(LineSeries::new(
                    times
                        .iter()
                        .zip(positions.iter())
                        .map(|(&x, &y)| (x, y))
                        .collect::<Vec<_>>(),
                    style,
                ))?
                .label(config.legend_label.clone())
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));
        } else {
            // 使用普通线图绘制
            chart
                .draw_series(LineSeries::new(
                    times
                        .iter()
                        .zip(positions.iter())
                        .map(|(&x, &y)| (x, y))
                        .collect::<Vec<_>>(),
                    style,
                ))?
                .label(config.legend_label.clone())
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));

            // 如果需要显示点
            if config.show_points {
                chart.draw_series(
                    times
                        .iter()
                        .zip(positions.iter())
                        .map(|(&x, &y)| Circle::new((x, y), config.point_size, style))
                        .collect::<Vec<_>>(),
                )?;
            }
        }

        Ok(())
    }
}

impl<P: ContinuousProcess> Plotter for ContinuousPlotter<P> {
    fn plot(&self, config: &PlotConfig) -> XResult<()> {
        // Get data
        let (times, positions) = self.trajectory.simulate(self.time_step)?;

        // Determine the coordinate range
        let x_range = if times.is_empty() {
            0.0..1.0
        } else {
            *times.first().unwrap_or(&0.0)..*times.last().unwrap_or(&1.0) * 1.05
        };

        let y_min = positions.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = positions.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_range = if y_min == y_max {
            y_min - 1.0..y_min + 1.0
        } else {
            let padding = (y_max - y_min) * 0.05;
            (y_min - padding)..(y_max + padding)
        };

        // Create the drawing backend
        let backend = create_backend(config)?;

        // Create the drawing area
        let root = backend.into_drawing_area();
        root.fill(&WHITE)?;

        // Create the chart
        let mut chart = ChartBuilder::on(&root)
            .caption(config.title.clone(), ("sans-serif", 20).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_range, y_range)?;

        // 绘制网格
        if config.show_grid {
            chart
                .configure_mesh()
                .x_labels(10)
                .y_labels(10)
                .x_desc(config.x_label.clone())
                .y_desc(config.y_label.clone())
                .draw()?;
        }

        // 绘制轨迹
        self.draw_single_trajectory(&mut chart, config)?;

        // 绘制图例
        if config.show_legend {
            chart
                .configure_series_labels()
                .border_style(&BLACK)
                .draw()?;
        }

        // 保存图表
        root.present()?;

        Ok(())
    }

    fn plot_multi(&self, config: &PlotConfig, num_trajectories: usize) -> XResult<()> {
        if num_trajectories == 0 {
            return Err(SimulationError::InvalidParameters(
                "num_trajectories must be positive".to_string(),
            )
            .into());
        }

        // 获取第一条轨迹以确定坐标范围
        let (times, positions) = self.trajectory.simulate(self.time_step)?;

        // 确定坐标范围
        let x_range = if times.is_empty() {
            0.0..1.0
        } else {
            *times.first().unwrap_or(&0.0)..*times.last().unwrap_or(&1.0) * 1.05
        };

        let mut y_min = positions.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let mut y_max = positions.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // 创建绘图后端
        let backend = create_backend(config)?;

        // 创建绘图区域
        let root = backend.into_drawing_area();
        root.fill(&WHITE)?;

        // 创建图表（暂时使用初始范围）
        let mut chart = ChartBuilder::on(&root)
            .caption(config.title.clone(), ("sans-serif", 20).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_range.clone(), y_min..y_max)?;

        // 绘制多条轨迹
        let colors = [
            "#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F", "#000000",
            "#FF0000", "#00FF00",
        ];

        // 绘制第一条轨迹
        let style = ShapeStyle {
            color: parse_color(&config.line_color)?,
            filled: false,
            stroke_width: config.line_width,
        };

        if config.use_step_plot {
            chart
                .draw_series(LineSeries::new(
                    times
                        .iter()
                        .zip(positions.iter())
                        .map(|(&x, &y)| (x, y))
                        .collect::<Vec<_>>(),
                    style,
                ))?
                .label(format!("{} 1", config.legend_label))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));
        } else {
            chart
                .draw_series(LineSeries::new(
                    times
                        .iter()
                        .zip(positions.iter())
                        .map(|(&x, &y)| (x, y))
                        .collect::<Vec<_>>(),
                    style,
                ))?
                .label(format!("{} 1", config.legend_label))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));
        }

        // 绘制额外的轨迹
        for i in 1..num_trajectories {
            let (times, positions) = self.trajectory.simulate(self.time_step)?;

            // 更新y轴范围
            let traj_y_min = positions.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let traj_y_max = positions.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            y_min = y_min.min(traj_y_min);
            y_max = y_max.max(traj_y_max);

            let color_idx = i % colors.len();
            let style = ShapeStyle {
                color: parse_color(colors[color_idx])?,
                filled: false,
                stroke_width: config.line_width,
            };

            if config.use_step_plot {
                chart
                    .draw_series(LineSeries::new(
                        times
                            .iter()
                            .zip(positions.iter())
                            .map(|(&x, &y)| (x, y))
                            .collect::<Vec<_>>(),
                        style,
                    ))?
                    .label(format!("{} {}", config.legend_label, i + 1))
                    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));
            } else {
                chart
                    .draw_series(LineSeries::new(
                        times
                            .iter()
                            .zip(positions.iter())
                            .map(|(&x, &y)| (x, y))
                            .collect::<Vec<_>>(),
                        style,
                    ))?
                    .label(format!("{} {}", config.legend_label, i + 1))
                    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));
            }
        }

        // 调整y轴范围
        let padding = (y_max - y_min) * 0.05;
        chart.set_y_range((y_min - padding), (y_max + padding));

        // 绘制网格
        if config.show_grid {
            chart
                .configure_mesh()
                .x_labels(10)
                .y_labels(10)
                .x_desc(config.x_label.clone())
                .y_desc(config.y_label.clone())
                .draw()?;
        }

        // 绘制图例
        if config.show_legend {
            chart
                .configure_series_labels()
                .border_style(&BLACK)
                .draw()?;
        }

        // 保存图表
        root.present()?;

        Ok(())
    }
}

/// Discrete process plotter
pub struct PointPlotter<P: PointProcess> {
    /// Trajectory object
    pub trajectory: PointTrajectory<P>,
}

impl<P: PointProcess> PointPlotter<P> {
    /// Create a new discrete process plotter
    pub fn new(trajectory: PointTrajectory<P>) -> Self {
        Self { trajectory }
    }

    /// Draw a single trajectory
    fn draw_single_trajectory<DB: DrawingBackend>(
        &self,
        chart: &mut ChartContext<DB, Cartesian2d<f64, f64>>,
        config: &PlotConfig,
    ) -> XResult<()> {
        let (times, positions) = if self.trajectory.duration.is_some() {
            self.trajectory.simulate_with_duration()?
        } else {
            self.trajectory.simulate_with_step()?
        };

        // Convert i64 to f64 for plotting
        let positions_f64: Vec<f64> = positions.iter().map(|&x| x as f64).collect();

        let style = ShapeStyle {
            color: parse_color(&config.line_color)?,
            filled: false,
            stroke_width: config.line_width,
        };

        // For discrete processes, default to using step plots
        chart
            .draw_series(LineSeries::new(
                times
                    .iter()
                    .zip(positions_f64.iter())
                    .map(|(&x, &y)| (x, y))
                    .collect::<Vec<_>>(),
                style,
            ))?
            .label(config.legend_label.clone())
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));

        // 如果需要显示点
        if config.show_points {
            chart.draw_series(
                times
                    .iter()
                    .zip(positions_f64.iter())
                    .map(|(&x, &y)| Circle::new((x, y), config.point_size, style))
                    .collect::<Vec<_>>(),
            )?;
        }

        Ok(())
    }
}

impl<P: PointProcess> Plotter for PointPlotter<P> {
    fn plot(&self, config: &PlotConfig) -> XResult<()> {
        // Get data
        let (times, positions) = if self.trajectory.duration.is_some() {
            self.trajectory.simulate_with_duration()?
        } else {
            self.trajectory.simulate_with_step()?
        };

        // Convert i64 to f64 for plotting
        let positions_f64: Vec<f64> = positions.iter().map(|&x| x as f64).collect();

        // Determine the coordinate range
        let x_range = if times.is_empty() {
            0.0..1.0
        } else {
            *times.first().unwrap_or(&0.0)..*times.last().unwrap_or(&1.0) * 1.05
        };

        let y_min = positions_f64.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = positions_f64
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_range = if y_min == y_max {
            y_min - 1.0..y_min + 1.0
        } else {
            let padding = (y_max - y_min) * 0.05;
            (y_min - padding)..(y_max + padding)
        };

        // Create the drawing backend
        let backend = create_backend(config)?;

        // Create the drawing area
        let root = backend.into_drawing_area();
        root.fill(&WHITE)?;

        // Create the chart
        let mut chart = ChartBuilder::on(&root)
            .caption(config.title.clone(), ("sans-serif", 20).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_range, y_range)?;

        // 绘制网格
        if config.show_grid {
            chart
                .configure_mesh()
                .x_labels(10)
                .y_labels(10)
                .x_desc(config.x_label.clone())
                .y_desc(config.y_label.clone())
                .draw()?;
        }

        // 绘制轨迹
        self.draw_single_trajectory(&mut chart, config)?;

        // 绘制图例
        if config.show_legend {
            chart
                .configure_series_labels()
                .border_style(&BLACK)
                .draw()?;
        }

        // 保存图表
        root.present()?;

        Ok(())
    }

    fn plot_multi(&self, config: &PlotConfig, num_trajectories: usize) -> XResult<()> {
        if num_trajectories == 0 {
            return Err(SimulationError::InvalidParameters(
                "num_trajectories must be positive".to_string(),
            )
            .into());
        }

        // 获取第一条轨迹以确定坐标范围
        let (times, positions) = if self.trajectory.duration.is_some() {
            self.trajectory.simulate_with_duration()?
        } else {
            self.trajectory.simulate_with_step()?
        };

        // 将i64转换为f64以便绘图
        let positions_f64: Vec<f64> = positions.iter().map(|&x| x as f64).collect();

        // 确定坐标范围
        let x_range = if times.is_empty() {
            0.0..1.0
        } else {
            *times.first().unwrap_or(&0.0)..*times.last().unwrap_or(&1.0) * 1.05
        };

        let mut y_min = positions_f64.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let mut y_max = positions_f64
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // 创建绘图后端
        let backend = create_backend(config)?;

        // 创建绘图区域
        let root = backend.into_drawing_area();
        root.fill(&WHITE)?;

        // 创建图表（暂时使用初始范围）
        let mut chart = ChartBuilder::on(&root)
            .caption(config.title.clone(), ("sans-serif", 20).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_range.clone(), y_min..y_max)?;

        // 绘制多条轨迹
        let colors = [
            "#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F", "#000000",
            "#FF0000", "#00FF00",
        ];

        // 绘制第一条轨迹
        let style = ShapeStyle {
            color: parse_color(&config.line_color)?,
            filled: false,
            stroke_width: config.line_width,
        };

        chart
            .draw_series(LineSeries::new(
                times
                    .iter()
                    .zip(positions_f64.iter())
                    .map(|(&x, &y)| (x, y))
                    .collect::<Vec<_>>(),
                style,
            ))?
            .label(format!("{} 1", config.legend_label))
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));

        // 绘制额外的轨迹
        for i in 1..num_trajectories {
            let (times, positions) = if self.trajectory.duration.is_some() {
                self.trajectory.simulate_with_duration()?
            } else {
                self.trajectory.simulate_with_step()?
            };

            // 将i64转换为f64以便绘图
            let positions_f64: Vec<f64> = positions.iter().map(|&x| x as f64).collect();

            // 更新y轴范围
            let traj_y_min = positions_f64.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let traj_y_max = positions_f64
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            y_min = y_min.min(traj_y_min);
            y_max = y_max.max(traj_y_max);

            let color_idx = i % colors.len();
            let style = ShapeStyle {
                color: parse_color(colors[color_idx])?,
                filled: false,
                stroke_width: config.line_width,
            };

            chart
                .draw_series(LineSeries::new(
                    times
                        .iter()
                        .zip(positions_f64.iter())
                        .map(|(&x, &y)| (x, y))
                        .collect::<Vec<_>>(),
                    style,
                ))?
                .label(format!("{} {}", config.legend_label, i + 1))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));
        }

        // 调整y轴范围
        let padding = (y_max - y_min) * 0.05;
        chart.set_y_range((y_min - padding), (y_max + padding));

        // 绘制网格
        if config.show_grid {
            chart
                .configure_mesh()
                .x_labels(10)
                .y_labels(10)
                .x_desc(config.x_label.clone())
                .y_desc(config.y_label.clone())
                .draw()?;
        }

        // 绘制图例
        if config.show_legend {
            chart
                .configure_series_labels()
                .border_style(&BLACK)
                .draw()?;
        }

        // 保存图表
        root.present()?;

        Ok(())
    }
}

/// Backend types supported by the plotter
#[derive(Debug)]
pub enum PlotterBackend {
    BitMap(BitMapBackend<'static>),
    SVG(SVGBackend<'static>),
}

impl DrawingBackend for PlotterBackend {
    type ErrorType = DrawError;

    fn get_size(&self) -> (u32, u32) {
        match self {
            PlotterBackend::BitMap(b) => b.get_size(),
            PlotterBackend::SVG(b) => b.get_size(),
        }
    }

    fn ensure_prepared(&mut self) -> Result<(), DrawError> {
        match self {
            PlotterBackend::BitMap(b) => b.ensure_prepared().map_err(DrawError::from),
            PlotterBackend::SVG(b) => b.ensure_prepared().map_err(DrawError::from),
        }
    }

    fn present(&mut self) -> Result<(), DrawError> {
        match self {
            PlotterBackend::BitMap(b) => b.present().map_err(DrawError::from),
            PlotterBackend::SVG(b) => b.present().map_err(DrawError::from),
        }
    }

    fn draw_pixel(&mut self, point: (i32, i32), color: RGBAColor) -> Result<(), DrawError> {
        match self {
            PlotterBackend::BitMap(b) => b.draw_pixel(point, color).map_err(DrawError::from),
            PlotterBackend::SVG(b) => b.draw_pixel(point, color).map_err(DrawError::from),
        }
    }

    fn draw_line<S: Color>(
        &mut self,
        from: (i32, i32),
        to: (i32, i32),
        style: &S,
    ) -> Result<(), DrawError> {
        match self {
            PlotterBackend::BitMap(b) => b.draw_line(from, to, style).map_err(DrawError::from),
            PlotterBackend::SVG(b) => b.draw_line(from, to, style).map_err(DrawError::from),
        }
    }

    fn estimate_text_size<S: std::string::ToString>(
        &self,
        text: &S,
        font: &FontDesc,
    ) -> Result<(u32, u32), DrawError> {
        match self {
            PlotterBackend::BitMap(b) => b.estimate_text_size(text, font).map_err(DrawError::from),
            PlotterBackend::SVG(b) => b.estimate_text_size(text, font).map_err(DrawError::from),
        }
    }

    fn draw_text<S: std::string::ToString>(
        &mut self,
        text: &S,
        style: &TextStyle,
        pos: (i32, i32),
    ) -> Result<(), DrawError> {
        match self {
            PlotterBackend::BitMap(b) => b.draw_text(text, style, pos).map_err(DrawError::from),
            PlotterBackend::SVG(b) => b.draw_text(text, style, pos).map_err(DrawError::from),
        }
    }
}

/// Create a drawing backend based on the configuration
fn create_backend(config: &PlotConfig) -> Result<BitMapBackend, DrawingErrorKind<DrawError>> {
    BitMapBackend::new(&config.output_path, (config.width, config.height))
        .map_err(|e| DrawingErrorKind::DrawingError(DrawError::from(e)))
}

/// Parse color string to RGB color
fn parse_color(color_str: &str) -> Result<RGBColor, DrawingErrorKind<DrawError>> {
    // 处理颜色名称
    match color_str.to_lowercase().as_str() {
        "red" => Ok(RED.into()),
        "blue" => Ok(BLUE.into()),
        "green" => Ok(GREEN.into()),
        "black" => Ok(BLACK.into()),
        "white" => Ok(WHITE.into()),
        "yellow" => Ok(YELLOW.into()),
        "cyan" => Ok(CYAN.into()),
        "magenta" => Ok(MAGENTA.into()),
        _ => {
            // 处理十六进制颜色代码
            if color_str.starts_with('#') && color_str.len() == 7 {
                let r = u8::from_str_radix(&color_str[1..3], 16)
                    .map_err(|e| DrawingErrorKind::DrawingError(DrawError::from(e)))?;
                let g = u8::from_str_radix(&color_str[3..5], 16)
                    .map_err(|e| DrawingErrorKind::DrawingError(DrawError::from(e)))?;
                let b = u8::from_str_radix(&color_str[5..7], 16)
                    .map_err(|e| DrawingErrorKind::DrawingError(DrawError::from(e)))?;
                Ok(RGBColor(r, g, b))
            } else {
                Err(DrawingErrorKind::DrawingError(DrawError::from(
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("Invalid color format: {}", color_str),
                    ),
                )))
            }
        }
    }
}

/// Create a plotter for continuous processes
pub fn plot_continuous<P: ContinuousProcess>(
    trajectory: ContinuousTrajectory<P>,
    time_step: f64,
    config: &PlotConfig,
) -> XResult<()> {
    let plotter = ContinuousPlotter::new(trajectory, time_step);
    if config.multi_trajectory {
        plotter.plot_multi(config, config.num_trajectories)
    } else {
        plotter.plot(config)
    }
}

/// Create a plotter for discrete processes
pub fn plot_point<P: PointProcess>(
    trajectory: PointTrajectory<P>,
    config: &PlotConfig,
) -> XResult<()> {
    let plotter = PointPlotter::new(trajectory);
    if config.multi_trajectory {
        plotter.plot_multi(config, config.num_trajectories)
    } else {
        plotter.plot(config)
    }
}

/// Convenient function: plot continuous process
pub fn plot_continuous_process<P: ContinuousProcess>(
    process: &P,
    duration: f64,
    time_step: f64,
    config: &PlotConfig,
) -> XResult<()> {
    let trajectory = process.duration(duration)?;
    plot_continuous(trajectory, time_step, config)
}

/// Convenient function: plot discrete process (using duration)
pub fn plot_point_process_with_duration<P: PointProcess>(
    process: &P,
    duration: f64,
    config: &PlotConfig,
) -> XResult<()> {
    let trajectory = process.duration(duration)?;
    plot_point(trajectory, config)
}

/// Convenient function: plot discrete process (using steps)
pub fn plot_point_process_with_steps<P: PointProcess>(
    process: &P,
    num_steps: usize,
    config: &PlotConfig,
) -> XResult<()> {
    let trajectory = process.step(num_steps)?;
    plot_point(trajectory, config)
}

/// Plot data directly
///
/// # Parameters
///
/// * `times` - Time series
/// * `positions` - Position series
/// * `config` - Plotting configuration
///
/// # Return value
///
/// Success or error
pub fn plot_data(times: &[f64], positions: &[f64], config: &PlotConfig) -> XResult<()> {
    if times.len() != positions.len() {
        return Err(SimulationError::InvalidInput(
            "Time series and position series length must be the same".to_string(),
        )
        .into());
    }

    // Create the drawing backend
    let backend = create_backend(config)?;

    // Create the drawing area
    let root = backend.into_drawing_area();
    root.fill(&WHITE)?;

    // Determine the coordinate range
    let x_range = if times.is_empty() {
        0.0..1.0
    } else {
        *times.first().unwrap_or(&0.0)..*times.last().unwrap_or(&1.0) * 1.05
    };

    let y_min = positions.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = positions.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_range = if y_min == y_max {
        y_min - 1.0..y_min + 1.0
    } else {
        let padding = (y_max - y_min) * 0.05;
        (y_min - padding)..(y_max + padding)
    };

    // Create the chart
    let mut chart = ChartBuilder::on(&root)
        .caption(config.title.clone(), ("sans-serif", 20).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_range, y_range)?;

    // Draw the grid
    if config.show_grid {
        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc(config.x_label.clone())
            .y_desc(config.y_label.clone())
            .draw()?;
    }

    // Draw the data
    let color = parse_color(&config.line_color)?;
    let style = ShapeStyle {
        color: color.mix(1.0),
        filled: false,
        stroke_width: config.line_width as u32,
    };

    if config.use_step_plot {
        // Use step plot
        let points: Vec<_> = times
            .iter()
            .zip(positions.iter())
            .map(|(&x, &y)| (x, y))
            .collect();

        chart.draw_series(
            points
                .iter()
                .map(|&(x, y)| Circle::new((x, y), config.point_size, style)),
        )?;

        chart
            .draw_series(LineSeries::new(points, style))?
            .label(config.legend_label.clone())
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));
    } else {
        // Use normal line plot
        let points: Vec<_> = times
            .iter()
            .zip(positions.iter())
            .map(|(&x, &y)| (x, y))
            .collect();

        chart
            .draw_series(LineSeries::new(points.clone(), style))?
            .label(config.legend_label.clone())
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));

        // If you need to display points
        if config.show_points {
            chart.draw_series(
                points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), config.point_size, style)),
            )?;
        }
    }

    // 绘制图例
    if config.show_legend {
        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
    }

    // 保存图表
    root.present()?;

    Ok(())
}

/// Plot multiple trajectories directly through simulated data
///
/// # Parameters
///
/// * `data` - Multiple trajectories data, each trajectory is a (times, positions) tuple
/// * `config` - Plotting configuration
///
/// # Return value
///
/// Success or error
pub fn plot_multi_data(data: &[(&[f64], &[f64])], config: &PlotConfig) -> XResult<()> {
    if data.is_empty() {
        return Err(SimulationError::InvalidParameters(
            "Trajectory data cannot be empty".to_string(),
        )
        .into());
    }

    for (i, (times, positions)) in data.iter().enumerate() {
        if times.len() != positions.len() {
            return Err(SimulationError::InvalidParameters(format!(
                "Trajectory {} time series and position series length must be the same",
                i
            ))
            .into());
        }

        if times.is_empty() {
            return Err(SimulationError::InvalidParameters(format!(
                "Trajectory {} time series and position series cannot be empty",
                i
            ))
            .into());
        }
    }

    // 确定所有轨迹的坐标范围
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for (times, positions) in data.iter() {
        x_min = x_min.min(*times.first().unwrap());
        x_max = x_max.max(*times.last().unwrap());

        let traj_y_min = positions.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let traj_y_max = positions.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        y_min = y_min.min(traj_y_min);
        y_max = y_max.max(traj_y_max);
    }

    let x_range = x_min..x_max * 1.05;
    let y_padding = (y_max - y_min) * 0.05;
    let y_range = (y_min - y_padding)..(y_max + y_padding);

    // 创建绘图后端
    let backend = create_backend(config)?;

    // 创建绘图区域
    let root = backend.into_drawing_area();
    root.fill(&WHITE)?;

    // 创建图表
    let mut chart = ChartBuilder::on(&root)
        .caption(config.title.clone(), ("sans-serif", 20).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_range, y_range)?;

    // 绘制网格
    if config.show_grid {
        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc(config.x_label.clone())
            .y_desc(config.y_label.clone())
            .draw()?;
    }

    // 绘制多条轨迹
    let colors = [
        "#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F", "#000000",
        "#FF0000", "#00FF00",
    ];

    for (i, (times, positions)) in data.iter().enumerate() {
        let color_idx = i % colors.len();
        let color = if i == 0 && !config.line_color.is_empty() {
            config.line_color.clone()
        } else {
            colors[color_idx].to_string()
        };

        let style = ShapeStyle {
            color: parse_color(&color)?,
            filled: false,
            stroke_width: config.line_width,
        };

        if config.use_step_plot {
            // 使用阶梯图绘制
            chart
                .draw_series(LineSeries::new(
                    times
                        .iter()
                        .zip(positions.iter())
                        .map(|(&x, &y)| (x, y))
                        .collect::<Vec<_>>(),
                    style,
                ))?
                .label(format!("{} {}", config.legend_label, i + 1))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));
        } else {
            // 使用普通线图绘制
            chart
                .draw_series(LineSeries::new(
                    times
                        .iter()
                        .zip(positions.iter())
                        .map(|(&x, &y)| (x, y))
                        .collect::<Vec<_>>(),
                    style,
                ))?
                .label(format!("{} {}", config.legend_label, i + 1))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));
        }

        // 如果需要显示点
        if config.show_points {
            chart.draw_series(
                times
                    .iter()
                    .zip(positions.iter())
                    .map(|(&x, &y)| Circle::new((x, y), config.point_size, style))
                    .collect::<Vec<_>>(),
            )?;
        }
    }

    // 绘制图例
    if config.show_legend {
        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
    }

    // 保存图表
    root.present()?;

    Ok(())
}

/// Plot discrete process directly through simulated data
///
/// # Parameters
///
/// * `times` - Time series
/// * `positions` - Position series (integer)
/// * `config` - Plotting configuration
///
/// # Return value
///
/// Success or error
pub fn plot_point_data(times: &[f64], positions: &[i64], config: &PlotConfig) -> XResult<()> {
    if times.len() != positions.len() {
        return Err(SimulationError::InvalidParameters(
            "Time series and position series length must be the same".to_string(),
        )
        .into());
    }

    if times.is_empty() {
        return Err(SimulationError::InvalidParameters(
            "Time series and position series cannot be empty".to_string(),
        )
        .into());
    }

    // 将i64转换为f64以便绘图
    let positions_f64: Vec<f64> = positions.iter().map(|&x| x as f64).collect();

    // 确定坐标范围
    let x_range = *times.first().unwrap()..*times.last().unwrap() * 1.05;

    let y_min = positions_f64.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = positions_f64
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_range = if y_min == y_max {
        y_min - 1.0..y_min + 1.0
    } else {
        let padding = (y_max - y_min) * 0.05;
        (y_min - padding)..(y_max + padding)
    };

    // 创建绘图后端
    let backend = create_backend(config)?;

    // 创建绘图区域
    let root = backend.into_drawing_area();
    root.fill(&WHITE)?;

    // 创建图表
    let mut chart = ChartBuilder::on(&root)
        .caption(config.title.clone(), ("sans-serif", 20).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_range, y_range)?;

    // 绘制网格
    if config.show_grid {
        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc(config.x_label.clone())
            .y_desc(config.y_label.clone())
            .draw()?;
    }

    // 设置样式
    let style = ShapeStyle {
        color: parse_color(&config.line_color)?,
        filled: false,
        stroke_width: config.line_width,
    };

    // 对于离散过程，默认使用阶梯图
    chart
        .draw_series(LineSeries::new(
            times
                .iter()
                .zip(positions_f64.iter())
                .map(|(&x, &y)| (x, y))
                .collect::<Vec<_>>(),
            style,
        ))?
        .label(config.legend_label.clone())
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));

    // 如果需要显示点
    if config.show_points {
        chart.draw_series(
            times
                .iter()
                .zip(positions_f64.iter())
                .map(|(&x, &y)| Circle::new((x, y), config.point_size, style))
                .collect::<Vec<_>>(),
        )?;
    }

    // 绘制图例
    if config.show_legend {
        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
    }

    // 保存图表
    root.present()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_data() {
        let times = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let positions = vec![0, 1, 2, 3, 4, 5];
        let config = PlotConfig::builder().build().unwrap();
        plot_data(&times, &positions, &config).unwrap();
    }
}
