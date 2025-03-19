use diffusionx::{
    simulation::{Fbm, prelude::*},
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
        .output_path("fbm_h03.png")
        .title("Fractional Brownian Motion (H=0.3)")
        .x_label("Time t")
        .y_label("Position X(t)")
        .size((800, 600))
        .backend(PlotterBackend::BitMap)
        .build()
        .unwrap();

    traj1.plot(&config1).unwrap();
    println!("Fractional Brownian Motion (H=0.3) trajectory saved to fbm_h03.png");

    // Visualize the second trajectory (H=0.5)
    let config2 = PlotConfigBuilder::default()
        .time_step(dt)
        .output_path("fbm_h05.png")
        .title("Fractional Brownian Motion (H=0.5)")
        .x_label("Time t")
        .y_label("Position X(t)")
        .size((800, 600))
        .backend(PlotterBackend::BitMap)
        .build()
        .unwrap();

    traj2.plot(&config2).unwrap();
    println!("Fractional Brownian Motion (H=0.5) trajectory saved to fbm_h05.png");

    // Visualize the third trajectory (H=0.7)
    let config3 = PlotConfigBuilder::default()
        .time_step(dt)
        .output_path("fbm_h07.png")
        .title("Fractional Brownian Motion (H=0.7)")
        .x_label("Time t")
        .y_label("Position X(t)")
        .size((800, 600))
        .backend(PlotterBackend::BitMap)
        .build()
        .unwrap();

    traj3.plot(&config3).unwrap();
    println!("Fractional Brownian Motion (H=0.7) trajectory saved to fbm_h07.png");
}
