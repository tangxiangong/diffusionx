use diffusionx::random::{
    exponential::Exponential, normal::Normal, poisson::Poisson, stable::Stable,
};

fn main() {
    println!("===== Example of Basic Random Number Generation =====");

    // Example of Normal Distribution
    let normal = Normal::new(0.0, 1.0).unwrap();
    let normal_samples = normal.samples(10).unwrap();

    println!(
        "Normal Distribution (μ=0, σ=1) samples: {:?}",
        normal_samples
    );

    // Example of Exponential Distribution
    let exponential = Exponential::new(2.0).unwrap();
    let exponential_samples = exponential.samples(10);

    println!(
        "Exponential Distribution (λ=2) samples: {:?}",
        exponential_samples
    );

    // Example of Poisson Distribution
    let poisson = Poisson::new(5.0).unwrap();
    let poisson_samples = poisson.samples(10);

    println!("Poisson Distribution (λ=5) samples: {:?}", poisson_samples);

    // Example of Stable Distribution
    let stable = Stable::new(1.5, 0.5, 1.0, 0.0).unwrap();
    let stable_samples = stable.samples(10).unwrap();

    println!(
        "Stable Distribution (α=1.5, β=0.5, σ=1, μ=0) samples: {:?}",
        stable_samples
    );
}
