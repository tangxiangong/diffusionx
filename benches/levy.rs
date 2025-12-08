use criterion::{Criterion, criterion_group, criterion_main};
use diffusionx::simulation::{continuous::Levy, prelude::*};
use std::hint::black_box;

fn criterion_benchmark(_: &mut Criterion) {
    let levy = Levy::new(0.0f32, 1.7f32).unwrap();

    let duration = 100.0f32;
    let time_step = 0.01;
    let particles = 10_000;
    let order = 0.7f32;

    let mut criterion = Criterion::default().sample_size(10);

    criterion.bench_function("levy-frac-raw-moment-cpu-f32", |b| {
        b.iter(|| {
            let _ = levy
                .frac_raw_moment(
                    black_box(duration),
                    black_box(order),
                    black_box(particles),
                    black_box(time_step),
                )
                .unwrap();
        })
    });

    #[cfg(feature = "cuda")]
    criterion.bench_function("levy-frac-raw-moment-cuda-f32", |b| {
        b.iter(|| {
            let _ = levy
                .frac_raw_moment_gpu(
                    black_box(duration),
                    black_box(order),
                    black_box(particles),
                    black_box(time_step),
                )
                .unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
