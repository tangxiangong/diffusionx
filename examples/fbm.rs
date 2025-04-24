use diffusionx::{
    simulation::{continuous::Fbm, prelude::*},
    visualize::{PlotConfigBuilder, PlotterBackend, Visualize},
};

fn main() {
    println!("===== Example of Fractional Brownian Motion Simulation =====");

    // Create different Hurst index fractional Brownian motions
    // Hurst index H = 0.3 (anti-persistent process)
    let fbm1 = Fbm::new(0.0, 0.3).unwrap();

    // Hurst index H = 0.5 (standard Brownian motion)
    let fbm2 = Fbm::new(0.0, 0.5).unwrap();

    // Hurst index H = 0.7 (persistent/long-range correlated process)
    let fbm3 = Fbm::new(0.0, 0.7).unwrap();

    // Time settings
    let t_max = 10.0; // Maximum time
    let dt = 0.01; // Time step

    // Generate trajectories
    let traj1 = fbm1.duration(t_max).unwrap();
    let traj2 = fbm2.duration(t_max).unwrap();
    let traj3 = fbm3.duration(t_max).unwrap();

    // Visualize the first trajectory (H=0.3)
    let config1 = PlotConfigBuilder::default()
        .time_step(dt)
        .output_path("tmp/fbm_h03.svg")
        .caption("Fractional Brownian Motion (H=0.3)")
        .x_label("t")
        .y_label("X(t)")
        .legend("fbm_h03")
        .size((800, 600))
        .backend(PlotterBackend::SVG)
        .build()
        .unwrap();

    traj1.plot(&config1).unwrap();
    println!("Fractional Brownian Motion (H=0.3) trajectory saved to tmp/fbm_h03.svg");

    // Visualize the second trajectory (H=0.5)
    let config2 = PlotConfigBuilder::default()
        .time_step(dt)
        .output_path("tmp/fbm_h05.svg")
        .caption("Fractional Brownian Motion (H=0.5)")
        .x_label("t")
        .y_label("X(t)")
        .legend("fbm_h05")
        .size((800, 600))
        .backend(PlotterBackend::SVG)
        .build()
        .unwrap();

    traj2.plot(&config2).unwrap();
    println!("Fractional Brownian Motion (H=0.5) trajectory saved to tmp/fbm_h05.svg");

    // Visualize the third trajectory (H=0.7)
    let config3 = PlotConfigBuilder::default()
        .time_step(dt)
        .output_path("tmp/fbm_h07.svg")
        .caption("Fractional Brownian Motion (H=0.7)")
        .x_label("t")
        .y_label("X(t)")
        .legend("fbm_h07")
        .size((800, 600))
        .backend(PlotterBackend::SVG)
        .build()
        .unwrap();

    traj3.plot(&config3).unwrap();
    println!("Fractional Brownian Motion (H=0.7) trajectory saved to tmp/fbm_h07.svg");
}
