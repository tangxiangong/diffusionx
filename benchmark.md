# Benchmark

生成长度为 `100_000_000` 的随机数组
|  | 标准正态分布 | `[0, 1]` 均匀分布 |
| :---: | :---: | :---: |
| [DiffusionX (Rust)](./diffusionx/)  | 273.85 ms | 245.78 ms |
| [DiffusionX (Python)](./py-diffusionx/) | 310 ms | 252 ms |
| [Julia](https://julialang.org/) | 581.61 ms | 371.37 ms |
| [NumPy](https://numpy.org/) | 3.28 s | 1.15 s |
| [Octave](https://octave.org/) | 1.31 s | 1.01 s |
| [Baltamatica](https://www.baltamatica.com/) | 5.47 s | 1.09 s |
