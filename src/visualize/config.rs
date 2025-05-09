use crate::{XResult, simulation::prelude::*, utils::minmax};
use derive_builder::Builder;
use plotters::{prelude::*, style::Color as _};
use std::{ops::Range, path::PathBuf};

pub use plotters::prelude::FontStyle;

/// Line Style
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum LineStyle {
    /// Solid
    #[default]
    Solid,
    /// Dashed
    Dashed,
    /// Dotted
    Dotted,
}

/// Plotters Backend
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PlotterBackend {
    #[default]
    BitMap,
    SVG,
}

/// Color
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Color {
    /// Red
    Red,
    /// Blue
    Blue,
    /// Green
    Green,
    /// Black
    Black,
    /// White
    White,
    /// Yellow
    Yellow,
    /// Cyan
    Cyan,
    /// Magenta
    Magenta,
    /// RGB color
    RGB(u8, u8, u8),
}

impl From<Color> for RGBColor {
    fn from(color: Color) -> Self {
        match color {
            Color::Red => RED,
            Color::Blue => BLUE,
            Color::Green => GREEN,
            Color::Black => BLACK,
            Color::White => WHITE,
            Color::Yellow => YELLOW,
            Color::Cyan => CYAN,
            Color::Magenta => MAGENTA,
            Color::RGB(r, g, b) => RGBColor(r, g, b),
        }
    }
}

#[allow(dead_code)]
/// Configuration for plotting
#[derive(Builder, Clone)]
#[builder(pattern = "mutable")]
pub struct PlotConfig {
    /// Backend
    #[builder(default)]
    pub(crate) backend: PlotterBackend,

    /// Background color
    #[builder(default = "Color::White")]
    pub(crate) background_color: Color,

    /// Title
    #[builder(default = "\"\".into()", setter(into))]
    pub(crate) title: String,

    /// Font family of the title
    #[builder(default = "sans-serif".into(), setter(into))]
    pub(crate) title_font_family: String,

    /// Font size of the title
    #[builder(default = "50.0", setter(into))]
    pub(crate) title_font_size: f64,

    /// Font style of the title
    #[builder(default = "FontStyle::Normal")]
    pub(crate) title_font_style: FontStyle,

    /// Caption
    #[builder(default = "\"Trajectory\".into()", setter(into))]
    pub(crate) caption: String,

    /// Font family of the caption
    #[builder(default = "sans-serif".into(), setter(into))]
    pub(crate) caption_font_family: String,

    /// Font size of the caption
    #[builder(default = "30.0", setter(into))]
    pub(crate) caption_font_size: f64,

    /// Font style of the caption
    #[builder(default = "FontStyle::Normal")]
    pub(crate) caption_font_style: FontStyle,

    /// The desired size of the four chart margins in backend units (pixels).
    #[builder(default = "5", setter(into))]
    pub(crate) margin: u32,

    /// The desired size of the X label area in backend units (pixels). If set to 0, the X label area is removed.
    #[builder(default = "30", setter(into))]
    pub(crate) x_label_area_size: u32,

    /// The desired size of the Y label area in backend units (pixels). If set to 0, the Y label area is removed.
    #[builder(default = "30", setter(into))]
    pub(crate) y_label_area_size: u32,

    /// Specifies the X axis range and data properties
    #[builder(setter(into, strip_option), default)]
    pub(crate) x_spec: Option<Range<f64>>,

    /// Specifies the Y axis range and data properties
    #[builder(setter(into, strip_option), default)]
    pub(crate) y_spec: Option<Range<f64>>,

    /// X-axis label
    #[builder(default = "String::from(\"Time\")", setter(into))]
    pub(crate) x_label: String,

    /// Y-axis label
    #[builder(default = "String::from(\"Position\")", setter(into))]
    pub(crate) y_label: String,

    /// Time step
    #[builder(default = "0.01", setter(into))]
    pub(crate) time_step: f64,

    /// Size (width, height) of the plot (pixels)
    #[builder(default = "(800, 600)", setter(into))]
    pub(crate) size: (u32, u32),

    /// Output file path
    #[builder(default = "PathBuf::from(\"result.png\")", setter(into))]
    pub(crate) output_path: PathBuf,

    /// Whether to show grid lines
    #[builder(default = "true")]
    pub(crate) show_grid: bool,

    /// Line color
    #[builder(default = "Color::Blue")]
    pub(crate) line_color: Color,

    /// Line style
    #[builder(default)]
    pub(crate) line_style: LineStyle,

    /// Whether to show legend
    #[builder(default = "true")]
    pub(crate) show_legend: bool,

    /// Legend title
    #[builder(default = "\"Trajectory\".into()", setter(into))]
    pub(crate) legend: String,

    /// Whether to show points
    #[builder(default = "false")]
    pub(crate) show_points: bool,

    /// Size of the points
    #[builder(default = "3", setter(into))]
    pub(crate) point_size: u32,

    /// Whether to fill the points with color
    #[builder(default = "false")]
    pub(crate) filled: bool,

    /// Dash style pattern [dash_length, spacing, stroke_width]
    #[builder(default = "[5, 10, 1]", setter(into))]
    pub(crate) dash_style: [u32; 3],

    /// Dot style pattern
    /// size/stroke_width: The size/stroke_width of the marker
    /// spacing: The spacing between markers
    #[builder(default = "[1, 1]", setter(into))]
    pub(crate) dot_style: [u32; 2],
}

impl PlotConfig {
    /// Plot the continuous trajectory
    pub(crate) fn plot<Backend: DrawingBackend, Process: ContinuousProcess>(
        &self,
        backend: Backend,
        traj: &ContinuousTrajectory<Process>,
    ) -> XResult<()> {
        let (times, positions) = traj.simulate(self.time_step)?;
        let max_time = *times.last().unwrap();
        let (min_x, max_x) = minmax(&positions);
        let meta = (max_time, min_x, max_x);
        let points: Vec<(f64, f64)> = times.iter().zip(positions).map(|(&t, x)| (t, x)).collect();
        set_config(self, backend, points, meta)
    }

    /// Plot the stair trajectory
    pub(crate) fn stair<Backend: DrawingBackend, Process: PointProcess>(
        &self,
        backend: Backend,
        traj: &PointTrajectory<Process>,
    ) -> XResult<()> {
        let (times, positions) = traj.simulate_with_duration()?;
        let max_time = *times.last().unwrap();
        let (min_x, max_x) = minmax(&positions);
        let meta = (max_time, min_x, max_x);
        let points: Vec<(f64, f64)> = times
            .iter()
            .zip(positions)
            .enumerate()
            .flat_map(|(i, (&t, y))| {
                if i == times.len() - 1 {
                    vec![(t, y)]
                } else {
                    vec![(t, y), (times[i + 1], y)]
                }
            })
            .collect();
        set_config(self, backend, points, meta)
    }
}

/// Set the configuration for the plot
pub(crate) fn set_config<Backend: DrawingBackend>(
    config: &PlotConfig,
    backend: Backend,
    data: Vec<(f64, f64)>,
    meta: (f64, f64, f64),
) -> XResult<()> {
    let (max_time, min_x, max_x) = meta;

    let x_spec = match config.x_spec.clone() {
        Some(x_spec) => x_spec,
        None => 0.0..max_time,
    };

    let y_spec = match config.y_spec.clone() {
        Some(y_spec) => y_spec,
        None => {
            let min_x = min_x * 1.25;
            let max_x = max_x * 1.25;
            min_x..max_x
        }
    };

    // Title font
    let title_font_familiy = config.title_font_family.as_str().into();
    let title_font = FontDesc::new(
        title_font_familiy,
        config.title_font_size,
        config.title_font_style,
    );
    let root = backend.into_drawing_area();
    let root = root.titled(&config.title, title_font)?;

    // Background color
    let background_color: RGBColor = config.background_color.clone().into();
    root.fill(&background_color)?;

    // Caption font
    let caption_font_familiy = config.caption_font_family.as_str().into();
    let caption_font = FontDesc::new(
        caption_font_familiy,
        config.caption_font_size,
        config.caption_font_style,
    );
    let mut chart = ChartBuilder::on(&root)
        .caption(&config.caption, caption_font)
        .margin(config.margin)
        .x_label_area_size(config.x_label_area_size)
        .y_label_area_size(config.y_label_area_size)
        .build_cartesian_2d(x_spec, y_spec)?;

    if config.show_grid {
        chart
            .configure_mesh()
            .x_desc(&config.x_label)
            .y_desc(&config.y_label)
            .draw()?;
    } else {
        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc(&config.x_label)
            .y_desc(&config.y_label)
            .draw()?;
    };

    let line_color: RGBColor = config.line_color.clone().into();
    let legend_color = line_color;

    let dash_shape_style = ShapeStyle {
        color: line_color.into(),
        filled: config.filled,
        stroke_width: config.dash_style[2],
    };

    let dot_shape_style = ShapeStyle {
        color: line_color.into(),
        filled: config.filled,
        stroke_width: config.dot_style[0],
    };

    let tmp = match config.line_style {
        LineStyle::Solid => {
            let line = if config.show_points {
                if config.filled {
                    LineSeries::new(data, line_color.filled()).point_size(config.point_size)
                } else {
                    LineSeries::new(data, line_color).point_size(config.point_size)
                }
            } else {
                LineSeries::new(data, line_color)
            };
            chart.draw_series(line)?
        }
        LineStyle::Dashed => {
            let line = DashedLineSeries::new(
                data,
                config.dash_style[0],
                config.dash_style[1],
                dash_shape_style,
            );
            chart.draw_series(line)?
        }
        LineStyle::Dotted => {
            let line = DashedLineSeries::new(
                data,
                config.dot_style[0],
                config.dot_style[1],
                dot_shape_style,
            );
            chart.draw_series(line)?
        }
    };

    tmp.label(&config.legend);

    if config.show_legend {
        tmp.legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], legend_color));
    }

    chart
        .configure_series_labels()
        .background_style(background_color)
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_config_builder() {
        let config = PlotConfigBuilder::default()
            .title("Test Plot")
            .x_label("Time")
            .y_label("Position")
            .size((800, 600))
            .output_path("test_plot.png")
            .build()
            .unwrap();

        assert_eq!(config.title, "Test Plot");
        assert_eq!(config.x_label, "Time");
        assert_eq!(config.y_label, "Position");
        assert_eq!(config.size, (800, 600));
        assert_eq!(config.output_path, PathBuf::from("test_plot.png"));
    }
}
