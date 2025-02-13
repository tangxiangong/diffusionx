use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use diffusionx::random;

const N: usize = 100_000_000;

fn criterion_benchmark(c: &mut Criterion) {
    
    c.bench_function("normal random number generation", |b| {
        b.iter(|| random::normal::standard_rands(black_box(N)));
    });

    c.bench_function("uniform random number generation", |b| {
        b.iter(|| random::uniform::standard_rands(black_box(N)));
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
