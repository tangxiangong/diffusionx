# DiffusionX

## Benchmarks
样本数 `100_000_000`
|  | 标准正态分布 | `[0, 1]` 均匀分布 |
| :---: | :---: | :---: |
| diffusionx (Rust)  | 273.85 ms | 245.78 ms |
| diffusionx (Python) | 310 ms | 252 ms |
| NumPy | 3.28 s | 1.15 s |
| Random.jl (Julia) | 581.61 ms | 371.37 ms |
