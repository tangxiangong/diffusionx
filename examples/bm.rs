use diffusionx::{
    simulation::{Bm, prelude::*},
    visualize::{PlotConfigBuilder, PlotterBackend, Visualize},
};

fn main() {
    println!("===== Example of Brownian Motion Simulation =====");

    // Create Brownian motion object
    let bm = Bm::default();

    // Time settings
    let t_max = 10.0; // Maximum time
    let dt = 0.01; // Time step

    // Generate Brownian motion trajectory
    let (times, positions) = bm.simulate(t_max, dt).unwrap();

    // Print some data points
    println!("Time\tPosition");
    for i in (0..times.len()).step_by(100) {
        println!("{:.2}\t{:.6}", times[i], positions[i]);
    }

    // Create trajectory and visualize
    let traj = bm.duration(t_max).unwrap();

    // Visualize trajectory
    let config = PlotConfigBuilder::default()
        .time_step(dt)
        .output_path("brownian_motion.png")
        .title("Brownian Motion Trajectory")
        .x_label("Time t")
        .y_label("Position X(t)")
        .size((800, 600))
        .backend(PlotterBackend::BitMap)
        .build()
        .unwrap();

    traj.plot(&config).unwrap();
    println!("Trajectory image saved to brownian_motion.png");
}
