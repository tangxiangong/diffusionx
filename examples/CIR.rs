use std::ops::AddAssign;

#[cfg(feature = "io")]
use diffusionx::utils::write_csv;
use diffusionx::{XError, XResult, random::normal, simulation::prelude::*};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

/// Cox-Ingersoll-Ross (CIR) process
///
/// $$ dX_t = \kappa (\theta - X_t) dt + \sigma \sqrt{X_t} dW_t $$
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
struct CIR<T: Float = f64> {
    /// speed of mean reversion
    kappa: T,
    /// long-term mean
    theta: T,
    /// volatility
    sigma: T,
    /// initial position
    init: T,
}

impl<T: Float> CIR<T> {
    fn new(kappa: T, theta: T, sigma: T, init: T) -> XResult<Self>
    where
        T: std::fmt::Debug,
    {
        if kappa <= T::zero() {
            return Err(XError::InvalidParameters(format!(
                "The parameter `kappa` must be positive, but got {kappa:?}"
            )));
        }
        Ok(Self {
            kappa,
            theta,
            sigma,
            init,
        })
    }
}

impl<T: std::fmt::Debug + Float + Send + Sync + AddAssign<T>> ContinuousProcess<T> for CIR<T>
where
    StandardNormal: Distribution<T>,
{
    fn start(&self) -> T {
        self.init
    }

    fn simulate(&self, duration: T, time_step: T) -> XResult<(Vec<T>, Vec<T>)> {
        if duration <= T::zero() {
            return Err(XError::InvalidParameters(format!(
                "The `duration` must be positive, got {duration:?}"
            )));
        }
        if time_step <= T::zero() {
            return Err(XError::InvalidParameters(format!(
                "The `time_step` must be positive, got `{time_step:?}`"
            )));
        }
        if duration < time_step {
            return Err(XError::InvalidParameters(format!(
                "The `duration` must be larger than `time_step`, got duration: {duration:?}, time_step: {time_step:?}"
            )));
        }

        let num_steps = ((duration / time_step).ceil()).to_usize().unwrap();

        let mut t = Vec::with_capacity(num_steps + 1);
        let mut x = Vec::with_capacity(num_steps + 1);

        let noises = normal::standard_rands::<T>(num_steps - 1);

        let mut current_t = T::zero();
        let mut current_x = self.init.max(T::zero());

        t.push(current_t);
        x.push(current_x);

        let mut drift;
        let mut diffusivity;
        let mut scale = self.sigma * time_step.sqrt();

        for xi in noises {
            drift = self.kappa * (self.theta - current_x);
            diffusivity = scale * current_x.sqrt();
            current_x += drift * time_step + diffusivity * xi;
            current_x = current_x.max(T::zero());
            current_t += time_step;

            t.push(current_t);
            x.push(current_x);
        }

        let last_step = duration - current_t;
        scale = self.sigma * last_step.sqrt();
        drift = self.kappa * (self.theta - current_x);
        diffusivity = scale * current_x.sqrt();
        current_x += drift * last_step + diffusivity * normal::standard_rand::<T>();
        current_x = current_x.max(T::zero());

        t.push(duration);
        x.push(current_x);

        Ok((t, x))
    }

    fn displacement(&self, duration: T, time_step: T) -> XResult<T> {
        if duration <= T::zero() {
            return Err(XError::InvalidParameters(format!(
                "The `duration` must be positive, got {duration:?}"
            )));
        }
        if time_step <= T::zero() {
            return Err(XError::InvalidParameters(format!(
                "The `time_step` must be positive, got `{time_step:?}`"
            )));
        }
        if duration < time_step {
            return Err(XError::InvalidParameters(format!(
                "The `duration` must be larger than `time_step`, got duration: {duration:?}, time_step: {time_step:?}"
            )));
        }

        let num_steps = ((duration / time_step).ceil()).to_usize().unwrap();

        let noises = normal::standard_rands::<T>(num_steps - 1);

        let mut current_t = T::zero();
        let mut current_x = self.init.max(T::zero());

        let mut drift;
        let mut diffusivity;
        let mut scale = self.sigma * time_step.sqrt();

        for xi in noises {
            drift = self.kappa * (self.theta - current_x);
            diffusivity = scale * current_x.sqrt();
            current_x += drift * time_step + diffusivity * xi;
            current_x = current_x.max(T::zero());
            current_t += time_step;
        }

        let last_step = duration - current_t;
        scale = self.sigma * last_step.sqrt();
        drift = self.kappa * (self.theta - current_x);
        diffusivity = scale * current_x.sqrt();
        current_x += drift * last_step + diffusivity * normal::standard_rand::<T>();
        current_x = current_x.max(T::zero());

        Ok(current_x - self.init)
    }
}

fn main() -> XResult<()> {
    let duration = 10.0;
    let particles = 10_000;
    let time_step = 0.01;
    let cir = CIR::new(1.0, 1.0, 1.0, 0.5)?;

    #[allow(unused)]
    let (t, x) = cir.simulate(duration, time_step)?;
    #[cfg(feature = "io")]
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

    #[cfg(feature = "visualize")]
    {
        let traj = cir.duration(duration)?;
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
    }
    Ok(())
}
