use diffusionx::{langevin, simulation::prelude::*};

const N: usize = 10_000;
const TAU: f64 = 0.01;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let f = |x: f64, _: f64| -x;
    let g = |_: f64, _: f64| 1.0;
    let eq = langevin!(dx = f(x, t)dt + g(x, t)dB(t), x(0) = 0.0)?;
    let now = std::time::Instant::now();
    let msds = (100..=1000)
        .step_by(100)
        .map(|t| eq.msd(t as f64, N, TAU).unwrap())
        .collect::<Vec<_>>();
    let elapsed = now.elapsed().as_secs_f64();
    println!("MSDs: {msds:#?}, elapsed time: {elapsed:.3} s");
    Ok(())
}
