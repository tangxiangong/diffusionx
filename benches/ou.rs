use criterion::{Criterion, criterion_group, criterion_main};
use diffusionx::simulation::{continuous::OrnsteinUhlenbeck as OU, prelude::*};
use std::hint::black_box;

fn criterion_benchmark(c: &mut Criterion) {
    let ou = OU::default();

    let duration = 100.0f32;
    let time_step = 0.01;
    let particles = 10_000;

    c.bench_function("ou-msd-cpu-f32", |b| {
        b.iter(|| {
            let _ = ou
                .msd(
                    black_box(duration),
                    black_box(particles),
                    black_box(time_step),
                )
                .unwrap();
        })
    });

    #[cfg(feature = "cuda")]
    c.bench_function("ou-msd-cuda-f32", |b| {
        b.iter(|| {
            let _ = ou
                .msd_gpu(
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
