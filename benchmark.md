# Benchmark

## 测试环境

### 硬件配置
- 设备型号：MacBook Pro 13-inch (2020)
- 处理器：Intel Core i5-1038NG7 @ 2.0GHz (4核8线程)
- 内存：16GB LPDDR4X 3733MHz
- 存储：NVMe SSD

### 软件环境
- 操作系统：macOS Sequoia 15.3
- Rust：1.85.0-beta.7
- Python：3.12
- Julia：1.11
- NumPy：2.2.2
- SciPy：1.15.1

## 测试结果

生成长度为 `10_000_000` 的随机数组

|  | 标准正态分布 | `[0, 1]` 均匀分布 | 稳定分布 |
| :---: | :---: | :---: | :---: |
| [DiffusionX (Rust)](./diffusionx/)  | 23.811 ms | 20.450 ms | 273.68 ms |
| [DiffusionX (Python)](./py-diffusionx/) | 24.1 ms | 21.687 ms | 277.6 ms |
| [Julia](https://julialang.org/) / [StableDistributions](https://github.com/jaksle/StableDistributions.jl) | 28.748 ms | 9.748 ms | 1.661 s |
| [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) | 295 ms | 81.2 ms | 3.39 s |
| [Numba](https://numba.pydata.org/) | - | - | 1.52 s |