use criterion::{Criterion, criterion_group, criterion_main};
use diffusionx::random::{exponential, normal, stable, uniform};
use std::hint::black_box;

const N: usize = 1 << 20;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("uniform distribution f64", |b| {
        b.iter(|| {
            let _ = uniform::standard_rands::<f64>(black_box(N));
        })
    });

    c.bench_function("uniform distribution f32", |b| {
        b.iter(|| {
            let _ = uniform::standard_rands::<f32>(black_box(N));
        })
    });

    #[cfg(feature = "cuda")]
    c.bench_function("uniform distribution f32 (CUDA)", |b| {
        b.iter(|| {
            let _ = diffusionx::gpu::random::curands(black_box(N)).unwrap();
        })
    });

    c.bench_function("normal distribution f64", |b| {
        b.iter(|| {
            let _ = normal::standard_rands::<f64>(black_box(N));
        })
    });

    c.bench_function("normal distribution f32", |b| {
        b.iter(|| {
            let _ = normal::standard_rands::<f64>(black_box(N));
        })
    });

    #[cfg(feature = "cuda")]
    c.bench_function("normal distribution f32 (cuda)", |b| {
        b.iter(|| {
            let _ = diffusionx::gpu::random::curandn(black_box(N), black_box(0.0), black_box(1.0))
                .unwrap();
        })
    });

    c.bench_function("exponential distribution f64", |b| {
        b.iter(|| {
            let _ = exponential::standard_rands::<f64>(black_box(N));
        })
    });

    c.bench_function("exponential distribution f32", |b| {
        b.iter(|| {
            let _ = exponential::standard_rands::<f32>(black_box(N));
        })
    });

    #[cfg(feature = "cuda")]
    c.bench_function("exponential distribution f32 (cuda)", |b| {
        b.iter(|| {
            let _ = diffusionx::gpu::random::curandexp(black_box(N)).unwrap();
        })
    });

    c.bench_function("stable distribution f64", |b| {
        b.iter(|| {
            let _ = stable::sym_standard_rands(black_box(0.7), black_box(N)).unwrap();
        })
    });

    c.bench_function("stable distribution f32", |b| {
        b.iter(|| {
            let _ = stable::sym_standard_rands(black_box(0.7f32), black_box(N)).unwrap();
        })
    });

    #[cfg(feature = "cuda")]
    c.bench_function("stable distribution f32 (cuda)", |b| {
        b.iter(|| {
            let _ = diffusionx::gpu::random::standard_stable_rands(
                black_box(0.7),
                black_box(0.0),
                black_box(N),
            )
            .unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
