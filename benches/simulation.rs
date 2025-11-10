use criterion::{Criterion, criterion_group, criterion_main};
use diffusionx::{langevin, simulation::prelude::*};
use std::hint::black_box;

fn criterion_benchmark(c: &mut Criterion) {
    let f = |x: f64, _: f64| -x;
    let g = |_: f64, _: f64| 1.0;
    let eq = langevin!(dx = f(x, t)dt + g(x, t)dB(t), x(0) = 0.0).unwrap();
    let duration = 1000.0;
    let time_step = 0.01;
    let particles = 10_000;
    let mut criterion = Criterion::default().sample_size(10);
    c.bench_function("langevin-simulation", |b| {
        b.iter(|| {
            let _ = eq
                .simulate(
                    black_box(duration),
                    // black_box(particles),
                    black_box(time_step),
                )
                .unwrap();
        })
    });

    criterion.bench_function("langevin-msd", |b| {
        b.iter(|| {
            let _ = eq
                .msd(
                    black_box(duration),
                    black_box(particles),
                    black_box(time_step),
                )
                .unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
