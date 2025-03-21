use diffusionx::{
    simulation::{continuous::Levy, prelude::*},
    visualize::{PlotConfigBuilder, PlotterBackend, Visualize},
};

fn main() {
    println!("===== Lévy process simulation example =====");

    // Create Lévy process object, alpha=1.5 represents the stable distribution index
    let levy = Levy::new(0.0, 1.5).unwrap();

    // Time settings
    let t_max = 10.0; // Maximum time
    let dt = 0.01; // Time step

    // Generate Lévy process trajectory
    let (times, positions) = levy.simulate(t_max, dt).unwrap();

    // Print some data points
    println!("Time\tPosition");
    for i in (0..times.len()).step_by(100) {
        println!("{:.2}\t{:.6}", times[i], positions[i]);
    }

    // Create trajectory and visualize
    let traj = levy.duration(t_max).unwrap();

    // Visualize trajectory
    let config = PlotConfigBuilder::default()
        .time_step(dt)
        .output_path("levy_process.png")
        .title("Lévy process trajectory")
        .x_label("Time t")
        .y_label("Position X(t)")
        .size((800, 600))
        .backend(PlotterBackend::BitMap)
        .build()
        .unwrap();

    traj.plot(&config).unwrap();
    println!("Trajectory image saved to levy_process.png");
}
