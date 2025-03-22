use diffusionx::{
    simulation::{continuous::OrnsteinUhlenbeck, prelude::*},
    visualize::{PlotConfigBuilder, PlotterBackend, Visualize},
};

fn main() {
    println!("===== Ornstein-Uhlenbeck process simulation example =====");

    // Create Ornstein-Uhlenbeck process object
    // Parameters: regression rate theta, volatility sigma, starting position
    let ou = OrnsteinUhlenbeck::new(0.5, 0.2, 1.0).unwrap();

    // Time settings
    let t_max = 20.0; // Maximum time
    let dt = 0.01; // Time step

    // Generate Ornstein-Uhlenbeck process trajectory
    let (times, positions) = ou.simulate(t_max, dt).unwrap();

    // Print some data points
    println!("Time\tPosition");
    for i in (0..times.len()).step_by(200) {
        println!("{:.2}\t{:.6}", times[i], positions[i]);
    }

    // Create trajectory and visualize
    let traj = ou.duration(t_max).unwrap();

    // Visualize trajectory
    let config = PlotConfigBuilder::default()
        .time_step(dt)
        .output_path("ornstein_uhlenbeck.png")
        .title("Ornstein-Uhlenbeck process trajectory")
        .x_label("Time t")
        .y_label("Position X(t)")
        .size((800, 600))
        .backend(PlotterBackend::BitMap)
        .build()
        .unwrap();

    traj.plot(&config).unwrap();
    println!("Trajectory image saved to ornstein_uhlenbeck.png");
}
