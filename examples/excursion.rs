use diffusionx::{
    simulation::{continuous::BrownianExcursion, prelude::*},
    visualize::{PlotConfigBuilder, PlotterBackend, Visualize},
};

fn main() {
    println!("===== Example of Brownian Excursion Simulation =====");

    // Create Brownian bridge object
    let be = BrownianExcursion;

    // Time settings
    let dt = 0.01; // Time step

    // Generate Brownian bridge trajectory
    let (times, positions) = be.simulate(1.0, dt).unwrap();

    // Print some data points
    println!("Time\tPosition");
    for i in (0..times.len()).step_by(100) {
        println!("{:.2}\t{:.6}", times[i], positions[i]);
    }

    // Create trajectory and visualize
    let traj = be.duration(1.0).unwrap();

    // Visualize trajectory
    let config = PlotConfigBuilder::default()
        .show_grid(false)
        .time_step(dt)
        .output_path("tmp/be.svg")
        .caption("Brownian Excursion Trajectory")
        .x_label("t")
        .y_label("X(t)")
        .legend("be")
        .size((800, 600))
        .backend(PlotterBackend::SVG)
        .build()
        .unwrap();

    traj.plot(&config).unwrap();
    println!("Trajectory image saved to tmp/be.svg");
}
