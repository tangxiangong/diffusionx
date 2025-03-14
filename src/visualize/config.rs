use derive_builder::Builder;
use std::path::PathBuf;

/// Configuration for plotting
#[derive(Builder, Clone, Debug)]
#[builder(setter(into))]
pub struct PlotConfig {
    /// Title of the plot
    #[builder(default = "String::from(\"Random Process Simulation\")", setter(into))]
    pub title: String,

    /// X-axis label
    #[builder(default = "String::from(\"Time\")", setter(into))]
    pub x_label: String,

    /// Y-axis label
    #[builder(default = "String::from(\"Position\")", setter(into))]
    pub y_label: String,

    /// Width of the plot (pixels)
    #[builder(default = "800", setter(into))]
    pub width: i32,

    /// Height of the plot (pixels)
    #[builder(default = "600", setter(into))]
    pub height: i32,

    /// Output file path
    #[builder(default = "PathBuf::from(\"output.png\")", setter(into))]
    pub output_path: PathBuf,

    /// Whether to show grid lines
    #[builder(default = "true")]
    pub show_grid: bool,

    /// Line width
    #[builder(default = "1.5", setter(into))]
    pub line_width: f64,

    /// Line color (RGB format, e.g. "#FF0000" for red, or color name like "red", "blue", etc.)
    #[builder(default = "String::from(\"#0072BD\")", setter(into))]
    pub line_color: String,

    /// Whether to use step plot (suitable for CTRW, Poisson, etc.)
    #[builder(default = "false")]
    pub use_step_plot: bool,

    /// Whether to show legend
    #[builder(default = "true")]
    pub show_legend: bool,

    /// Legend title
    #[builder(default = "String::from(\"Trajectory\")", setter(into))]
    pub legend_label: String,

    /// Whether to show points
    #[builder(default = "false")]
    pub show_points: bool,

    /// Size of the points
    #[builder(default = "3.0")]
    pub point_size: f64,
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
            .width(800)
            .height(600)
            .output_path("test_plot.png")
            .build()
            .unwrap();

        assert_eq!(config.title, "Test Plot");
        assert_eq!(config.x_label, "Time");
        assert_eq!(config.y_label, "Position");
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert_eq!(config.output_path, PathBuf::from("test_plot.png"));
    }
}
