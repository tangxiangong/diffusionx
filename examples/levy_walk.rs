use diffusionx::simulation::{continuous::LevyWalk, prelude::*};

fn main() {
    println!("===== Example of Levy Walk Simulation =====");

    // Create Brownian motion object
    let walk = LevyWalk::default();

    // Time settings
    let duration = 10.0; // Maximum time
    let time_step = 0.1; // Time step

    // Create trajectory and visualize
    let traj = walk.duration(duration).unwrap();

    let (_times, positions) = traj.simulate(time_step).unwrap();
    println!("positions: {positions:#?}");

    #[cfg(feature = "visualize")]
    {
        // Visualize trajectory
        let config = PlotConfigBuilder::default()
            .time_step(time_step)
            .output_path("tmp/levy_walk.svg")
            .caption("Levy Walk Trajectory")
            .x_label("t")
            .y_label("X")
            .legend("Levy Walk")
            .size((800, 600))
            .backend(PlotterBackend::SVG)
            .build()
            .unwrap();

        traj.plot(&config).unwrap();
        println!("Trajectory image saved to tmp/levy_walk.svg");
    }
}
