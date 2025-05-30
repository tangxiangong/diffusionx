use diffusionx::{
    XError, XResult,
    random::normal,
    simulation::prelude::*,
    utils::{diff, linspace, write_csv},
};

/// CIR
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
struct CIR {
    speed: f64,
    mean: f64,
    volatility: f64,
    start_position: f64,
}

impl CIR {
    fn new(
        speed: impl Into<f64>,
        mean: impl Into<f64>,
        volatility: impl Into<f64>,
        start_position: impl Into<f64>,
    ) -> XResult<Self> {
        let speed: f64 = speed.into();
        if speed <= 0.0 {
            return Err(XError::InvalidParameters(format!(
                "speed must be greater than 0, but got {}",
                speed
            )));
        }
        Ok(Self {
            speed,
            mean: mean.into(),
            volatility: volatility.into(),
            start_position: start_position.into(),
        })
    }
}

impl ContinuousProcess for CIR {
    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        let t = linspace(0.0, duration, time_step);
        let num_steps = t.len() - 1;
        let initial_x = self.start_position.max(0.0);
        let noises = normal::standard_rands::<f64>(num_steps);
        let delta = diff(&t);

        let x = std::iter::once(initial_x)
            .chain(
                noises
                    .iter()
                    .zip(delta)
                    .scan(initial_x, |state, (&xi, delta_t)| {
                        let current_x = *state;
                        let drift = self.speed * (self.mean - current_x);
                        let diffusion = self.volatility * current_x.sqrt().max(0.0);

                        let next_x = current_x + drift * delta_t + diffusion * xi * delta_t.sqrt();
                        *state = next_x.max(0.0);

                        Some(*state)
                    }),
            )
            .collect();

        Ok((t, x))
    }
}

fn main() -> XResult<()> {
    let duration = 10.0;
    let particles = 10_000;
    let time_step = 0.01;
    let cir = CIR::new(1, 1, 1, 0.5)?;
    let traj = cir.duration(duration)?;
    let (t, x) = cir.simulate(duration, time_step)?;
    write_csv("tmp/CIR.csv", &t, &x)?;
    // mean
    let mean = cir.mean(duration, particles, time_step)?; // or let mean = traj.raw_moment(1, particles, time_step)?;
    println!("mean: {mean}");
    // msd
    let msd = cir.msd(duration, particles, time_step)?; // or let msd = traj.central_moment(2, particles, time_step)?;
    println!("MSD: {msd}");
    // FPT
    let max_duration = 1000.0;
    let fpt = cir
        .fpt((-1.0, 1.0), max_duration, time_step)?
        .unwrap_or(-1.0);
    println!("FPT: {fpt}");
    // occupation time
    let occupation_time = cir.occupation_time((-1.0, 1.0), duration, time_step)?;
    println!("Occupation Time: {occupation_time}");
    // TAMSD
    let slag = 1.0;
    let quad_order = 10;
    let tamsd = TAMSD::new(&cir, duration, slag)?;
    let eatamsd = tamsd.mean(particles, time_step, quad_order)?;
    println!("EATAMSD: {eatamsd}");

    // Visualization
    let config = PlotConfigBuilder::default()
        .time_step(time_step)
        .output_path("tmp/CIR.svg")
        .caption("CIR")
        .show_grid(false)
        .x_label("t")
        .y_label("r")
        .legend("CIR")
        .backend(PlotterBackend::SVG)
        .build()
        .unwrap();
    traj.plot(&config)?;
    Ok(())
}
