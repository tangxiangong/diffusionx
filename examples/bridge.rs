use diffusionx::{
    simulation::{continuous::BrownianBridge, prelude::*},
    visualize::{PlotConfigBuilder, PlotterBackend, Visualize},
};

fn main() {
    println!("===== Example of Brownian Bridge Simulation =====");

    // Create Brownian bridge object
    let bb = BrownianBridge;

    // Time settings
    let t_max = 10.0; // Maximum time
    let dt = 0.01; // Time step

    // Generate Brownian bridge trajectory
    let (times, positions) = bb.simulate(t_max, dt).unwrap();

    // Print some data points
    println!("Time\tPosition");
    for i in (0..times.len()).step_by(100) {
        println!("{:.2}\t{:.6}", times[i], positions[i]);
    }

    // Create trajectory and visualize
    let traj = bb.duration(t_max).unwrap();

    // Visualize trajectory
    let config = PlotConfigBuilder::default()
        .show_grid(false)
        .time_step(dt)
        .output_path("tmp/bb.svg")
        .caption("Brownian Bridge Trajectory")
        .x_label("t")
        .y_label("X(t)")
        .legend("bb")
        .size((800, 600))
        .backend(PlotterBackend::SVG)
        .build()
        .unwrap();

    traj.plot(&config).unwrap();
    println!("Trajectory image saved to tmp/bb.svg");
}
