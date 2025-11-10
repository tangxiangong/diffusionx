use diffusionx::simulation::{continuous::Levy, prelude::*};

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

    #[cfg(feature = "visualize")]
    {
        // Create trajectory and visualize
        let traj = levy.duration(t_max).unwrap();

        // Visualize trajectory
        let config = PlotConfigBuilder::default()
            .time_step(dt)
            .output_path("tmp/levy.svg")
            .caption("Lévy process trajectory")
            .x_label("t")
            .y_label("X(t)")
            .legend("levy")
            .size((800, 600))
            .backend(PlotterBackend::SVG)
            .build()
            .unwrap();

        traj.plot(&config).unwrap();
        println!("Trajectory image saved to tmp/levy.svg");
    }
}
