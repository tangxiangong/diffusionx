use diffusionx::{
    simulation::{continuous::Bm, prelude::*},
    visualize::{PlotConfigBuilder, PlotterBackend, Visualize},
};

#[test]
fn test_visualize() {
    // Create Brownian motion object
    let bm = Bm::default();

    // Time settings
    let t_max = 10.0; // Maximum time
    let dt = 0.01; // Time step

    // Create trajectory and visualize
    let traj = bm.duration(t_max).unwrap();

    // Visualize trajectory
    let config = PlotConfigBuilder::default()
        .time_step(dt)
        .output_path("tmp/bm.svg")
        .caption("Brownian Motion")
        .x_label("t")
        .y_label("X(t)")
        .legend("bm")
        .show_grid(true)
        .size((800, 600))
        .backend(PlotterBackend::SVG)
        .build()
        .unwrap();

    traj.plot(&config).unwrap();
}
