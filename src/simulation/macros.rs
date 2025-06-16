/// Create a Langevin equation with a given drift and diffusion functions.
///
/// # Example
/// ```
/// use diffusionx::simulation::macros::langevin;
///
/// let f = |x: f64, t: f64| x * t;
/// let g = |x: f64, t: f64| x * t;
/// let equation = langevin!(dx = f(x, t)dt + g(x, t)dB, x(0) = 1.0);
/// ```
#[macro_export]
macro_rules! langevin {
    (dx = $drift:ident(x, t)dt + $diffusion: ident(x, t)dB, x(0)=$x0:expr) => {{
        use diffusionx::simulation::continuous::Langevin;
        let equation =
            Langevin::new($drift, $diffusion, $x0).expect("Failed to create Langevin equation");
        equation
    }};
}
