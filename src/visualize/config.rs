use crate::{XResult, simulation::prelude::*, utils::minmax};
use derive_builder::Builder;
use plotters::prelude::*;
use std::{ops::Range, path::PathBuf};

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
pub struct PlotConfig<'a> {
    /// Backend
    #[builder(default)]
    pub(crate) backend: PlotterBackend,

    /// Background color
    #[builder(default = "Color::White")]
    pub(crate) background_color: Color,

    /// Caption
    #[builder(default = "\"\".into()", setter(into))]
    pub(crate) caption: String,

    /// Font
    #[builder(default = "(\"sans-serif\", 10).into_font()", setter(into))]
    pub(crate) font: FontDesc<'a>,

    /// The desired size of the four chart margins in backend units (pixels).
    #[builder(default = "5", setter(into))]
    pub(crate) margin: u32,

    /// Title
    #[builder(default = "String::from(\"Random Process Simulation\")", setter(into))]
    pub(crate) title: String,

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

    /// Line width
    #[builder(default = "1.5", setter(into))]
    pub(crate) line_width: f64,

    /// Line color
    #[builder(default = "Color::Blue")]
    pub(crate) line_color: Color,

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
    #[builder(default = "3.0", setter(into))]
    pub(crate) point_size: f64,
}

impl PlotConfig<'_> {
    pub(crate) fn plot<Backend: DrawingBackend, Process: ContinuousProcess>(
        &self,
        backend: Backend,
        traj: &ContinuousTrajectory<Process>,
    ) -> XResult<()> {
        let (times, positions) = traj.simulate(self.time_step)?;
        let max_time = *times.last().unwrap();
        let (min_x, max_x) = minmax(&positions);

        let x_spec = match self.x_spec.clone() {
            Some(x_spec) => x_spec,
            None => 0.0..max_time,
        };

        let y_spec = match self.y_spec.clone() {
            Some(y_spec) => y_spec,
            None => {
                let min_x = min_x * 1.25;
                let max_x = max_x * 1.25;
                min_x..max_x
            }
        };

        let root = backend.into_drawing_area();
        let background_color: RGBColor = self.background_color.clone().into();
        root.fill(&background_color)?;
        let mut chart = ChartBuilder::on(&root)
            .caption(&self.title, self.font.clone().into_font())
            .margin(self.margin)
            .x_label_area_size(self.x_label_area_size)
            .y_label_area_size(self.y_label_area_size)
            .build_cartesian_2d(x_spec, y_spec)?;

        if self.show_grid {
            chart.configure_mesh().draw()?;
        } else {
            chart.configure_mesh().disable_mesh().draw()?;
        }

        let line_color: RGBColor = self.line_color.clone().into();
        let legend_color = line_color;

        chart
            .draw_series(LineSeries::new(
                times.iter().zip(positions).map(|(t, x)| (*t, x)),
                &line_color,
            ))?
            .label(&self.caption)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], legend_color));

        chart
            .configure_series_labels()
            .background_style(background_color)
            .border_style(BLACK)
            .draw()?;

        root.present()?;

        Ok(())
    }

    // pub(crate) fn stairs<Backend: DrawingBackend, Process: PointProcess>(
    //     &self,
    //     backend: Backend,
    //     _traj: &PointTrajectory<Process>,
    // ) -> XResult<()> {
    //     let _root = backend.into_drawing_area();
    //     todo!()
    // }
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
