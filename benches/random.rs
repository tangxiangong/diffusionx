use criterion::{Criterion, criterion_group, criterion_main};
use diffusionx::random::{exponential, normal, stable, uniform};
use std::hint::black_box;

const N: usize = 10_000_000;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("uniform distribution", |b| {
        b.iter(|| {
            let _ = uniform::standard_rands(black_box(N));
        })
    });

    c.bench_function("normal distribution", |b| {
        b.iter(|| {
            let _ = normal::standard_rands::<f64>(black_box(N));
        })
    });

    c.bench_function("exponential distribution", |b| {
        b.iter(|| {
            let _ = exponential::standard_rands::<f64>(black_box(N));
        })
    });

    c.bench_function("stable distribution", |b| {
        b.iter(|| {
            let _ = stable::sym_standard_rands(black_box(0.7), black_box(N)).unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
