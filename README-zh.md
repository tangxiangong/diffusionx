# DiffusionX

[English](README.md) | 简体中文

> DiffusionX 是一个多线程高性能的 Rust 随机数生成和随机过程模拟库。

[![文档](https://img.shields.io/badge/文档-最新-blue.svg)](https://docs.rs/diffusionx/latest/diffusionx/)
[![crates.io](https://img.shields.io/crates/v/diffusionx.svg)](https://crates.io/crates/diffusionx)
[![许可证: MIT/Apache-2.0](https://img.shields.io/badge/许可证-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

## 特性

- **高性能**：针对计算效率进行优化，通过 [rayon](https://github.com/rayon-rs/rayon) 支持多线程并行计算
- **可扩展**：基于 trait 的架构设计，便于扩展自定义过程和分布
- **文档完善**：详细的 API 文档，包含数学背景和使用示例
- **类型安全**：利用 Rust 的类型系统确保编译时安全和正确性
- **零成本抽象**：高效的抽象设计，最小化运行时开销

## 可视化

DiffusionX 使用 [plotters](https://crates.io/crates/plotters) 库提供内置的可视化功能：

- **过程轨迹**：轻松可视化连续过程轨迹
- **可定制化绘图**：配置绘图外观，包括颜色、尺寸和线条样式
- **多种输出格式**：支持位图和 SVG 输出格式
- **简洁 API**：基于 trait 的直观 API，便于可视化模拟结果

## 已实现功能

### 随机数生成

- [x] 正态分布
- [x] 均匀分布
- [x] 指数分布
- [x] 泊松分布
- [x] $\alpha$-稳定分布

### 随机过程

- [x] 布朗运动
- [x] $\alpha$-稳定莱维过程
- [x] 柯西过程
- [x] $\alpha$-稳定 Subordinator
- [x] 逆 $\alpha$-稳定逆 Subordinator
- [x] 泊松过程
- [x] 分数布朗运动
- [x] 连续时间随机游走
- [x] Ornstein-Uhlenbeck 过程
- [x] 朗之万方程
- [x] 广义朗之万方程
- [x] 从属朗之万方程
- [x] 莱维游走 - 具有耦合跳跃长度和等待时间的超扩散过程
- [x] 生灭过程
- [x] 随机游走
- [x] 布朗桥
- [x] Brownian excursion
- [x] Brownian meander
- [x] 伽马过程

## 安装

添加以下内容到您的 `Cargo.toml`:
```toml
[dependencies]
diffusionx = "*"  # 替换为最新版本
```

或者使用以下命令安装:
```bash
cargo add diffusionx
```

## 使用

### 随机数生成

```rust
use diffusionx::random::{normal, uniform, stable};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 生成一个均值为 0.0，标准差为 1.0 的正态随机数
    let normal_sample = normal::rand(0.0, 1.0)?;
    // 生成 1000 个标准正态随机数
    let std_normal_samples = normal::standard_rands(1000);

    // 生成一个 [0, 10) 范围内的均匀随机数
    let uniform_sample = uniform::range_rand(0..10)?;
    // 生成 1000 个 [0, 1) 范围内的均匀随机数
    let std_uniform_samples = uniform::standard_rands(1000);

    // 生成 1000 个标准稳定随机数
    let stable_samples = stable::standard_rands(1.5, 0.5, 1000)?;

    Ok(())
}
```

### 随机过程模拟

```rust
use diffusionx::simulation::{prelude::*, continuous::Bm};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建标准布朗运动对象
    let bm = Bm::default();
    // 创建持续时间为 1.0 的轨迹
    let traj = bm.duration(1.0)?;
    // 使用时间步长 0.01 模拟布朗运动轨迹
    let (times, positions) = traj.simulate(0.01)?;
    println!("times: {:?}", times);
    println!("positions: {:?}", positions);

    // 计算一阶原点矩，1000 个粒子，时间步长为 0.01
    let mean = traj.raw_moment(1, 1000, 0.01)?;
    println!("mean: {:?}", mean);
    // 计算二阶中心矩，1000 个粒子，时间步长为 0.01
    let msd = traj.central_moment(2, 1000, 0.01)?;
    println!("msd: {:?}", msd);
    // 计算 TAMSD，100.0 的持续时间，1.0 的 delta，10000 个粒子，时间步长为 0.1，Gauss-Legendre 积分阶数为 10
    let tamsd = bm.tamsd(100.0, 1.0, 10000, 0.1, 10)?;
    println!("tamsd: {:?}", tamsd);
    // 计算布朗运动首次通过时间，边界为 -1.0 和 1.0
    let fpt = bm.fpt((-1.0, 1.0), 1000, 0.01)?;
    println!("fpt: {:?}", fpt);
    Ok(())
}
```

### 可视化示例

```rust
use diffusionx::{
    simulation::{continuous::Bm, prelude::*},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建布朗运动轨迹
    let bm = Bm::default();
    let traj = bm.duration(10.0)?;

    // 配置并创建可视化
    let config = PlotConfigBuilder::default()
    .time_step(0.01)
    .output_path("brownian_motion.png")
    .caption("布朗运动轨迹")
    .x_label("t")
    .y_label("B")
    .legend("bm")
    .size((800, 600))
    .backend(PlotterBackend::BitMap)
    .build()?;

    // 生成图像
    traj.plot(&config)?;

    Ok(())
}
```

## 架构与可扩展性

DiffusionX 采用基于 trait 的系统设计，具有高度的可扩展性和性能优化：

### 核心 Trait

- `ContinuousProcess`: 连续随机过程的基本 trait
- `PointProcess`: 点过程的基本 trait
- `DiscreteProcess`: 离散随机过程的基本 trait
- `Moment`: 统计矩计算的 trait，包括原点矩和中心矩
- `Visualize`: 绘制过程轨迹的 trait

### 泛函分布模拟

DiffusionX 为随机过程提供强大的泛函分布模拟功能：

1. **首次通过时间 (FPT)**: 计算过程首次到达指定边界的时间
   ```rust
   // 对于布朗运动过程
   let bm = Bm::default();
   // 计算边界为 -1.0 和 1.0 的首次通过时间
   let fpt = bm.fpt(0.01, (-1.0, 1.0), 1000)?;
   ```

2. **占据时间**: 测量过程在指定区域内停留的时间
   ```rust
   // 对于布朗运动过程
   let bm = Bm::default();
   let traj = bm.duration(10.0)?;
   // 计算在区域 [0.0, 2.0] 内停留的时间
   let occupation = traj.occupation_time(0.01, (0.0, 2.0))?;
   ```

### 扩展自定义过程

1. 添加新的连续随机过程：
   ```rust
   #[derive(Clone)]
   struct MyProcess {
       // 您的参数应该是 `Send + Sync` 以支持并行计算
   }

   impl ContinuousProcess for MyProcess {
       fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<(Vec<f64>, Vec<f64>)> {
           todo!() // 实现您的模拟逻辑
       }
   }
   ```

2. 实现 `ContinuousProcess` trait 后自动实现
    - 均值 `mean(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64>`
    - 均方位移 `msd(&self, duration: impl Into<f64>, particles: usize, time_step: f64) -> XResult<f64>`
    - 原点矩 `raw_moment(&self, duration: impl Into<f64>, order: i32, particles: usize, time_step: f64) -> XResult<f64>`
    - 中心矩 `central_moment(&self, duration: impl Into<f64>, order: i32, particles: usize, time_step: f64) -> XResult<f64>`
    - 首次通过时间 `fpt(&self, domain: (impl Into<f64>, impl Into<f64>), max_duration: impl Into<f64>, time_step: f64) -> XResult<Option<f64>>`
    - 占据时间 `occupation_time(&self, domain: (impl Into<f64>, impl Into<f64>), duration: impl Into<f64>, time_step: f64) -> XResult<f64>`
    - 时间平均均方位移 `tamsd(&self, duration: impl Into<f64>, delta: impl Into<f64>, particles: usize, time_step: f64, quad_order: usize) -> XResult<f64>`
    - 路径 trait `ContinuousTrajectoryTrait` 和其用于可视化的子 trait `Visualize`

示例：
```rust
let myprocess = MyProcess::default();
let traj = myprocess.duration(10)?;
// 计算一阶原点矩，1000 个粒子，时间步长为 0.01
let mean = traj.raw_moment(1, 1000, 0.01)?;
```

1. 并行计算支持：
    - 矩计算自动支持通过 Rayon 进行并行计算
    - 统计量计算默认使用并行策略
    - 可配置的并行性能优化

2. 可视化支持：
    - 简单代码即可实现轨迹可视化
    - 高度可定制的绘图配置

示例：
```rust
// 可视化布朗运动轨迹
use diffusionx::visualize::{PlotConfigBuilder, Visualize};

let bm = Bm::default().duration(10)?;
let config = PlotConfigBuilder::default()
.title("布朗运动")
.output_path("brownian_motion.png")
.build()?;

bm.plot(&config)?;
```

## 基准测试
相关内容请见 [py-diffusionx](https://github.com/tangxiangong/py-diffusionx) 的 **基准测试** 部分。

## 许可证

本项目采用双许可证模式：

* [MIT 许可证](https://opensource.org/licenses/MIT)
* [Apache 许可证 2.0 版本](https://www.apache.org/licenses/LICENSE-2.0)

您可以选择使用其中任一许可证。
