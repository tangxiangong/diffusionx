use diffusionx::{
    XResult, generalized_langevin, langevin, simulation::prelude::*, subordinated_langevin,
};

fn main() -> XResult<()> {
    let f = |x: f64, t: f64| x * t.sqrt();
    let g = |x: f64, t: f64| x + t;
    // Create a Langevin equation with a given drift and diffusion functions.
    // dx = f(x, t)dt + g(x, t)dB, x(0) = 1.0
    let equation = langevin!(dx = f(x, t)dt + g(x, t)dB(t), x(0) = 1.0)?;
    let (t, x) = equation.simulate(1.0, 0.1)?;
    println!("t: {t:?}");
    println!("x: {x:?}");
    let equation = generalized_langevin!(dx = f(x, t)dt + g(x, t)dL(t), x(0) = 1.0, alpha = 1.5)?;
    let (t, x) = equation.simulate(1.0, 0.1)?;
    println!("t: {t:?}");
    println!("x: {x:?}");
    let equation =
        subordinated_langevin!(dx = f(x, t)dS(t) + g(x, t)dB(S(t)), x(0) = 1.0, alpha = 0.7)?;
    let (t, x) = equation.simulate(1.0, 0.1)?;
    println!("t: {t:?}");
    println!("x: {x:?}");
    Ok(())
}
