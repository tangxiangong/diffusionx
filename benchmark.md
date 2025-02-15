# Benchmark

生成长度为 `10_000_000` 的随机数组
|  | 标准正态分布 | `[0, 1]` 均匀分布 | 稳定分布 |
| :---: | :---: | :---: | :---: |
| [DiffusionX (Rust)](./diffusionx/)  | 23.811 ms | 20.450 ms | 273.68 ms |
| [DiffusionX (Python)](./py-diffusionx/) | 24.1 ms | 21.687 ms | 277.6 ms |
| [Julia](https://julialang.org/) / [StableDistributions](https://github.com/jaksle/StableDistributions.jl) | 28.748 ms | 9.748 ms | 1.661 s |
| [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) | 295 ms | 81.2 ms | 3.39 s |