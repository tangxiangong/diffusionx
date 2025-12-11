<div align=center>
<h1 aligh="center">
DiffusionX
</h1>
<p align="center">
一个多线程高性能的 Rust 随机数生成和随机过程模拟库，支持可选的 GPU 加速</p>
<p align="center">
English | <a href="README-zh.md">简体中文</a>
</p>
<p align="center">
<a href="https://crates.io/crates/diffusionx"> <img alt="Crates.io Version" src="https://img.shields.io/crates/v/diffusionx?style=for-the-badge"> </a>
<a href="https://docs.rs/diffusionx"> <img alt="docs.rs" src="https://img.shields.io/docsrs/diffusionx?style=for-the-badge"> </a>
<img alt="License: MIT OR Apache-2.0" src="https://img.shields.io/crates/l/diffusionx?style=for-the-badge">
<img alt="Downloads" src="https://img.shields.io/crates/d/diffusionx?style=for-the-badge">
</p>
</div>

## 已实现

### 随机数生成

> [!NOTE]
> DiffusionX 使用高质量的 [Xoshiro256++](https://prng.di.unimi.it/) 随机数生成器作为所有分布的公共熵源。

- 正态分布
- 均匀分布
- 指数分布
- 泊松分布
- $\alpha$-稳定分布

### 随机过程模拟

- 布朗运动
- $\alpha$-稳定 Lévy 过程
- 柯西过程
- $\alpha$-稳定从属过程
- 逆 $\alpha$-稳定从属过程
- 泊松过程
- 分数布朗运动
- 连续时间随机游走
- Ornstein-Uhlenbeck 过程
- Langevin 方程
- 广义 Langevin 方程
- 从属 Langevin 方程
- Lévy 行走
- 生灭过程
- 随机游走
- 布朗桥
- Brownian meander
- Brownian excursion
- 伽马过程
- 几何布朗运动
- 布朗非高斯过程

### GPU 加速 (CUDA/Metal)

- 布朗运动
- $\alpha$-稳定 Lévy 过程
- Ornstein-Uhlenbeck 过程
- $\alpha$-稳定分布随机数生成

## 快速开始

### 随机数生成

```rust
use diffusionx::random::{normal, uniform, stable};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // N(0, 1) 正态分布随机数
    let normal_sample = normal::rand(0.0, 1.0)?;
    // 生成 1000 个标准正态分布随机数
    let std_normal_samples = normal::standard_rands::<f64>(1000);

    // 生成范围在 [0, 10) 的均匀分布随机数
    let uniform_sample = uniform::range_rand(0..10)?;
    // 生成 1000 个范围在 [0, 1) 的均匀分布随机数
    let std_uniform_samples = uniform::standard_rands(1000);

    // 生成 1000 个标准 $\alpha$-稳定分布随机数
    let stable_samples = stable::standard_rands(1.5, 0.5, 1000)?;

    Ok(())
}
```

### 随机过程模拟

```rust
use diffusionx::simulation::{prelude::*, continuous::Bm};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bm = Bm::default();
    // 创建持续时间为 1.0 的轨迹
    let traj = bm.duration(1.0)?;
    // 以时间步长 0.01 模拟布朗运动轨迹
    let (times, positions) = traj.simulate(0.01)?;
    println!("times: {:?}", times);
    println!("positions: {:?}", positions);

    // 计算一阶原始矩，使用 1000 个粒子和时间步长 0.01
    let mean = traj.raw_moment(1, 1000, 0.01)?;
    println!("mean: {mean}");
    // 计算二阶中心矩，使用 1000 个粒子和时间步长 0.01
    let msd = traj.central_moment(2, 1000, 0.01)?;
    println!("MSD: {msd}");
    // 计算持续时间为 100.0，delta 为 1.0，10000 个粒子，时间步长 0.1，
    // 以及 Gauss-Legendre 求积阶数为 10 的 EATAMSD
    let eatamsd = bm.eatamsd(100.0, 1.0, 10000, 0.1, 10)?;
    println!("EATAMSD: {eatamsd}");
    // 计算布朗运动在边界 -1.0 和 1.0 处的首次通过时间
    let fpt = bm.fpt((-1.0, 1.0), 1000, 0.01)?;
    println!("fpt: {fpt}");
    Ok(())
}
```

### 可视化

> [!NOTE]
> 可视化功能需要启用 `visualize` 特性。
> ```toml
> # 在你的 Cargo.toml 中
> [dependencies]
> diffusionx = { version = "*", features = ["visualize"] }
> ```

```rust
use diffusionx::{
    simulation::{continuous::Bm, prelude::*},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建布朗运动轨迹
    let bm = Bm::default();
    let traj = bm.duration(10.0)?;

    // 配置绘图参数
    let config = PlotConfigBuilder::default()
    .time_step(0.01)
    .output_path("brownian_motion.png")
    .caption("Brownian Motion Trajectory")
    .x_label("t")
    .y_label("B")
    .legend("bm")
    .size((800, 600))
    .backend(PlotterBackend::BitMap)
    .build()?;

    // 生成绘图
    traj.plot(&config)?;

    Ok(())
}
```

### GPU 加速

> [!NOTE]
> 这需要启用 `metal` 或 `cuda` 特性。
> ```toml
> # 在你的 Cargo.toml 中
> [dependencies]
> diffusionx = { version = "*", features = ["cuda"] }
> ```

```rust
use diffusionx::{
    simulation::continuous::Bm,
    gpu::GPUMoment,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bm = Bm::<f32>::default();

    let mean = bm.mean_gpu(1.0, 100_000, 0.01)?;
    let msd = bm.msd_gpu(1.0, 100_000, 0.01)?;
    let raw_moment = bm.raw_moment_gpu(1.0, 2, 100_000, 0.01)?;
    let central_moment = bm.central_moment_gpu(1.0, 2, 100_000, 0.01)?;
    let frac_raw = bm.frac_raw_moment_gpu(1.0, 1.5, 100_000, 0.01)?;
    let frac_central = bm.frac_central_moment_gpu(1.0, 1.5, 100_000, 0.01)?;

    Ok(())
}
```

## 架构与可扩展性

DiffusionX 是基于 trait 的系统设计，具有高度的可扩展性和性能：

### 核心 Trait

- `ContinuousProcess`: 连续随机过程的基础 trait
- `PointProcess`: 点过程的基础 trait
- `DiscreteProcess`: 离散随机过程的基础 trait
- `Moment`: 统计矩计算的 trait，包括（分数阶）原始矩和中心矩
- `Visualize`: 用于绘制过程轨迹的 trait
- `GPUMoment`: 用于在 CUDA 中模拟（分数阶）矩的 trait。

`GPUMoment` trait 提供了 GPU 加速的统计矩计算。它被实现于：
- `Bm<T>` - 布朗运动
- `OrnsteinUhlenbeck<T>` - Ornstein-Uhlenbeck 过程
- `Levy<T>` - Lévy 过程

| 方法 | 描述 |
|--------|-------------|
| `mean_gpu(duration, particles, time_step)` | 计算均值（第一阶原始矩） |
| `msd_gpu(duration, particles, time_step)` | 计算均方位移（第二阶中心矩） |
| `raw_moment_gpu(duration, order, particles, time_step)` | 计算整数阶原始矩 |
| `central_moment_gpu(duration, order, particles, time_step)` | 计算整数阶中心矩 |
| `frac_raw_moment_gpu(duration, order, particles, time_step)` | 计算分数阶原始矩 |
| `frac_central_moment_gpu(duration, order, particles, time_step)` | 计算分数阶中心矩 |

### 扩展自定义过程

1. 添加新的连续过程：
   ```rust
   #[derive(Debug, Clone)]
   struct MyProcess {
       // 你的结构体字段应该是 `Send + Sync` 以支持并行计算
       // 并且实现 `Clone`
   }

   impl ContinuousProcess for MyProcess {
       fn start(&self) -> f64 {
           0.0 // 起点
       }

       fn simulate(
            &self,
            duration: f64,
            time_step: f64
        ) -> XResult<(Vec<f64>, Vec<f64>)> {
           // 实现你的模拟逻辑
           todo!()
       }
   }
   ```

2. 实现 `ContinuousProcess` trait 会自动提供以下功能：
    - 均值 `mean`
    - 均方位移 `msd`
    - (分数阶) 原始矩 `raw_moment` (`frac_raw_moment`)
    - (分数阶) 中心矩 `central_moment` (`frac_central_moment`)
    - 首次通过时间 `fpt`
    - 占用时间 `occupation_time`
    - TAMSD `tamsd`
    - 可视化 `plot`

完整的 CIR 过程实现示例见 [这里](./examples/CIR.rs)。

## 基准性能测试

性能基准测试比较了 Rust、C++、Julia 和 Python 的实现，测试结果可以在 [这里](https://github.com/tangxiangong/diffusionx-benches) 找到。

## 开源许可证

双重许可，您可以选择以下任一许可证：

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

### 贡献

除非你明确声明，否则你有意提交的任何贡献，
根据 Apache-2.0 许可证的定义，将按照上述双重许可进行授权，
且不附加任何额外条款或条件。
