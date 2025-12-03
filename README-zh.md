<div align=center>
<h1 aligh="center">
DiffusionX
</h1>
<p align="center">
一个多线程高性能的 Rust 随机数生成和随机过程模拟库
</p>
<p align="center">
<a href="README.md">English</a> | 简体中文
</p>
<p align="center">
<a href="https://crates.io/crates/diffusionx"> <img alt="Crates.io Version" src="https://img.shields.io/crates/v/diffusionx?style=for-the-badge"> </a>
<a href="https://docs.rs/diffusionx"> <img alt="docs.rs" src="https://img.shields.io/docsrs/diffusionx?style=for-the-badge"> </a>
<img alt="License: MIT OR Apache-2.0" src="https://img.shields.io/crates/l/diffusionx?style=for-the-badge">
<img alt="Downloads" src="https://img.shields.io/crates/d/diffusionx?style=for-the-badge">
</p>
</div>

## 已实现功能

### 随机数生成

- [x] 正态分布
- [x] 均匀分布
- [x] 指数分布
- [x] 泊松分布
- [x] $\alpha$-稳定分布

> [!NOTE]
> DiffusionX 在随机数生成模块中统一采用高质量的 [Xoshiro256++](https://prng.di.unimi.it/) 随机数发生器。

### 随机过程模拟

- [x] 布朗运动
- [x] $\alpha$-稳定莱维过程
- [x] 柯西过程
- [x] $\alpha$-稳定 subordinator
- [x] 逆 $\alpha$-稳定 subordinator
- [x] 泊松过程
- [x] 分数布朗运动
- [x] 连续时间随机游走
- [x] Ornstein-Uhlenbeck 过程
- [x] 朗之万方程
- [x] 广义朗之万方程
- [x] 从属朗之万方程
- [x] 莱维游走
- [x] 生灭过程
- [x] 随机游走
- [x] 布朗桥
- [x] Brownian excursion
- [x] Brownian meander
- [x] 伽马过程
- [x] 几何布朗运动
- [x] 布朗非高斯过程

## 使用

### 随机数生成

```rust
use diffusionx::random::{normal, uniform, stable};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 生成一个均值为 0.0，标准差为 1.0 的正态随机数
    let normal_sample = normal::rand(0.0, 1.0)?;
    // 生成 1000 个标准正态随机数
    let std_normal_samples = normal::standard_rands::<f64>(1000);

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
    println!("mean: {mean}");
    // 计算二阶中心矩，1000 个粒子，时间步长为 0.01
    let msd = traj.central_moment(2, 1000, 0.01)?;
    println!("MSD: {msd}");
    // 计算 EATAMSD，100.0 的持续时间，1.0 的 delta，10000 个粒子，时间步长为 0.1，
    // Gauss-Legendre 积分阶数为 10
    let eatamsd = bm.eatamsd(100.0, 1.0, 10000, 0.1, 10)?;
    println!("EATAMSD: {eatamsd}");
    // 计算布朗运动首次通过时间，边界为 -1.0 和 1.0
    let fpt = bm.fpt((-1.0, 1.0), 1000.0, 0.01)?;
    println!("FPT: {fpt}");
    Ok(())
}
```

### 可视化

> [!NOTE]
> 可视化功能需要开启 `visualize`
> ```toml
> # In your Cargo.toml
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

### 扩展自定义过程

1. 添加新的连续随机过程：
   ```rust
   #[derive(Debug, Clone)]
   struct MyProcess {
       // 您的参数应该是 `Send + Sync` 以支持并行计算, 并且可以需要实现 `Clone`
   }

   impl ContinuousProcess for MyProcess {
       fn start(&self) -> f64 {
           0.0  // 或者您希望的起始位置
       }

       fn simulate(
            &self,
            duration: f64,
            time_step: f64
        ) -> XResult<(Vec<f64>, Vec<f64>)> {
           todo!() // 实现您的模拟逻辑
       }
   }
   ```

2. 实现 `ContinuousProcess` trait 后自动实现
    - 均值 `mean`
    - 均方位移 `msd`
    - 原点矩 `raw_moment`
    - 中心矩 `central_moment`
    - 首次通过时间 `fpt`
    - 占据时间 `occupation_time`
    - 时间平均均方位移 `tamsd`
    - 可视化 `plot`

**示例：**

在您的项目目录中运行以下 Cargo 命令：
```bash
cargo add diffusionx --features io,visualize
```
或者在您的 Cargo.toml 中添加以下依赖：
```toml
[dependencies]
diffusionx = { version = "*", features = ["io", "visualize"] }
```

```rust
#[cfg(feature = "io")]
use diffusionx::utils::write_csv;
use diffusionx::{
    XError, XResult, check_duration_time_step,
    random::normal,
    simulation::prelude::*,
    utils::{diff, linspace},
};

/// CIR
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
struct CIR {
    speed: f64,
    mean: f64,
    volatility: f64,
    start_position: f64,
}

impl CIR {
    fn new(
        speed: impl Into<f64>,
        mean: impl Into<f64>,
        volatility: impl Into<f64>,
        start_position: impl Into<f64>,
    ) -> XResult<Self> {
        let speed: f64 = speed.into();
        if speed <= 0.0 {
            return Err(XError::InvalidParameters(format!(
                "speed must be greater than 0, but got {speed}"
            )));
        }
        Ok(Self {
            speed,
            mean: mean.into(),
            volatility: volatility.into(),
            start_position: start_position.into(),
        })
    }
}

impl ContinuousProcess for CIR {
    fn start(&self) -> f64 {
        self.start_position
    }

    fn simulate(&self, duration: f64, time_step: f64) -> XResult<Pair> {
        check_duration_time_step(duration, time_step)?;

        let t = linspace(0.0, duration, time_step);
        let num_steps = t.len() - 1;
        let initial_x = self.start_position.max(0.0);
        let noises = normal::standard_rands::<f64>(num_steps);
        let delta = diff(&t);

        let x = std::iter::once(initial_x)
            .chain(
                noises
                    .iter()
                    .zip(delta)
                    .scan(initial_x, |state, (&xi, delta_t)| {
                        let current_x = *state;
                        let drift = self.speed * (self.mean - current_x);
                        let diffusion = self.volatility * current_x.sqrt().max(0.0);

                        let next_x = current_x + drift * delta_t + diffusion * xi * delta_t.sqrt();
                        *state = next_x.max(0.0);

                        Some(*state)
                    }),
            )
            .collect();

        Ok((t, x))
    }
}

fn main() -> XResult<()> {
    let duration = 10.0;
    let particles = 10_000;
    let time_step = 0.01;
    let cir = CIR::new(1, 1, 1, 0.5)?;

    #[allow(unused)]
    let (t, x) = cir.simulate(duration, time_step)?;
    #[cfg(feature = "io")]
    write_csv("tmp/CIR.csv", &t, &x)?;
    // mean
    let mean = cir.mean(duration, particles, time_step)?; // or let mean = traj.raw_moment(1, particles, time_step)?;
    println!("mean: {mean}");
    // msd
    let msd = cir.msd(duration, particles, time_step)?; // or let msd = traj.central_moment(2, particles, time_step)?;
    println!("MSD: {msd}");
    // FPT
    let max_duration = 1000.0;
    let fpt = cir
        .fpt((-1.0, 1.0), max_duration, time_step)?
        .unwrap_or(-1.0);
    println!("FPT: {fpt}");
    // occupation time
    let occupation_time = cir.occupation_time((-1.0, 1.0), duration, time_step)?;
    println!("Occupation Time: {occupation_time}");
    // TAMSD
    let slag = 1.0;
    let quad_order = 10;
    let tamsd = TAMSD::new(&cir, duration, slag)?;
    let eatamsd = tamsd.mean(particles, time_step, quad_order)?;
    println!("EATAMSD: {eatamsd}");

    #[cfg(feature = "visualize")]
    {
        let traj = cir.duration(duration)?;
        // Visualization
        let config = PlotConfigBuilder::default()
            .time_step(time_step)
            .output_path("tmp/CIR.svg")
            .caption("CIR")
            .show_grid(false)
            .x_label("t")
            .y_label("r")
            .legend("CIR")
            .backend(PlotterBackend::SVG)
            .build()
            .unwrap();
        traj.plot(&config)?;
    }
    Ok(())
}
```

**结果：**
```
mean: 0.9957644815350275
MSD: 0.7441251895881059
FPT: 0.38
Occupation Time: 4.719999999999995
EATAMSD: 0.6085042089895467
```
<img src="https://raw.githubusercontent.com/tangxiangong/diffusionx/dev/assets/CIR.svg" alt="CIR"/>


## 基准测试

性能基准测试比较 Rust、C++、Julia 和 Python 实现的代码，请见[这里](https://github.com/tangxiangong/diffusionx-benches)。

## 许可协议

本项目采用以下任一许可协议：

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) 或 https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) 或 https://opensource.org/licenses/MIT)

您可以选择其中任意一种。

### 贡献

除非您明确声明，否则根据 Apache-2.0 许可协议的定义，您有意提交的任何贡献都将按照上述双重许可协议进行许可，不附加任何额外条款或条件。
