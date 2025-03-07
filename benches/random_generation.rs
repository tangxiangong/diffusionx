use criterion::{Criterion, criterion_group, criterion_main};
use diffusionx::random;
use std::hint::black_box;

const N: usize = 10_000_000;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("normal random number generation", |b| {
        b.iter(|| random::normal::standard_rands(black_box(N)));
    });

    c.bench_function("uniform random number generation", |b| {
        b.iter(|| random::uniform::standard_rands(black_box(N)));
    });

    c.bench_function("stable random number generation", |b| {
        b.iter(|| random::stable::skew_rands(0.4, black_box(N)));
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
