use diffusionx::{langevin, simulation::prelude::*};

fn main() {
    let f = |x: f64, t: f64| x * t.sqrt();
    let g = |x: f64, t: f64| x + t;
    // Create a Langevin equation with a given drift and diffusion functions.
    // dx = f(x, t)dt + g(x, t)dB, x(0) = 1.0
    let equation = langevin!(dx = f(x, t)dt + g(x, t)dB, x(0) = 1.0);
    let (t, x) = equation.simulate(1.0, 0.1).unwrap();
    println!("t: {:?}", t);
    println!("x: {:?}", x);
}
