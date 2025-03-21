use diffusionx::simulation::{jump::CTRW, prelude::*};

fn main() {
    println!("===== Example of CTRW Simulation =====");

    // Create CTRW instance
    // Parameters: alpha (jump index), beta (waiting time index), start_position (starting position)
    // alpha=0.8 represents a heavy-tailed jump distribution, beta=0.7 represents a long-tailed waiting time distribution
    let ctrw = CTRW::new(0.8, 0.7, 0.0).unwrap();

    // Time settings
    let t_max = 100.0; // Maximum time
    let dt = 0.1; // Time step

    // Generate CTRW trajectory
    // Here we call the simulate method provided by the ContinuousProcess trait
    // Note: The CTRW implementation does not actually use the time_step parameter
    let (times, positions) = ContinuousProcess::simulate(&ctrw, t_max, dt).unwrap();

    // 打印一些数据点
    println!("时间\t位置");
    for i in (0..times.len()).step_by(100) {
        if i < times.len() {
            println!("{:.2}\t{:.6}", times[i], positions[i]);
        }
    }
}
