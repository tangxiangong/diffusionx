# DiffusionX

## Benchmarks
生成长度为 `100_000_000` 的随机数组
|  | 标准正态分布 | `[0, 1]` 均匀分布 |
| :---: | :---: | :---: |
| [DiffusionX (Rust)](./diffusionx/)  | 273.85 ms | 245.78 ms |
| [DiffusionX (Python)](./py-diffusionx/) | 310 ms | 252 ms |
| [NumPy](https://numpy.org/) | 3.28 s | 1.15 s |
| [Random.jl (Julia)](https://github.com/JuliaRandom/Random.jl) | 581.61 ms | 371.37 ms |
| [Octave](https://www.gnu.org/software/octave/) | 1.31 s | 1.09 s |
|[Baltamatica](https://www.baltamatica.com/)| 5.47 s | 1.09 s |
