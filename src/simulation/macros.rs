/// Create a Langevin equation with a given drift and diffusion functions.
///
/// # Example
/// ```
/// use diffusionx::langevin;
///
/// let f = |x: f64, t: f64| x * t;
/// let g = |x: f64, t: f64| x * t;
/// let equation = langevin!(dx = f(x, t)dt + g(x, t)dB(t), x(0) = 1.0)?;
/// ```
#[macro_export]
macro_rules! langevin {
    (dx = $drift:ident(x, t)dt + $diffusion: ident(x, t)dB(t), x(0)=$x0:expr) => {{
        use diffusionx::simulation::continuous::Langevin;
        let equation = Langevin::new($drift, $diffusion, $x0);
        equation
    }};
}

/// Create a Generalized Langevin equation with a given drift and diffusion functions.
///
/// # Example
/// ```
/// use diffusionx::generalized_langevin;
///
/// let f = |x: f64, t: f64| x * t;
/// let g = |x: f64, t: f64| x * t;
/// let equation = generalized_langevin!(dx = f(x, t)dt + g(x, t)dL(t), x(0) = 1.0, alpha = 1.5)?;
/// ```
#[macro_export]
macro_rules! generalized_langevin {
    (dx = $drift:ident(x, t)dt + $diffusion: ident(x, t)dL(t), x(0)=$x0:expr, alpha = $alpha:expr) => {{
        use diffusionx::simulation::continuous::GeneralizedLangevin;
        let equation = GeneralizedLangevin::new($drift, $diffusion, $x0, $alpha);
        equation
    }};
}

/// Create a Subordinated Langevin equation with a given drift and diffusion functions.
///
/// # Example
/// ```
/// use diffusionx::subordinated_langevin;
///
/// let f = |x: f64, t: f64| x * t;
/// let g = |x: f64, t: f64| x * t;
/// let equation = subordinated_langevin!(dx = f(x, t) dS(t) + g(x, t) dB(S(t)), x(0) = 1.0, alpha = 0.7)?;
/// ```
#[macro_export]
macro_rules! subordinated_langevin {
    (dx = $drift:ident(x, t) dS(t) + $diffusion: ident(x, t) dB(S(t)), x(0)=$x0:expr, alpha = $alpha:expr) => {{
        use diffusionx::simulation::continuous::SubordinatedLangevin;
        let equation = SubordinatedLangevin::new($drift, $diffusion, $x0, $alpha);
        equation
    }};
}
