# DiffusionX

[English](README.md) | 简体中文

> [!NOTE]
> 开发进行中。DiffusionX 是一个多线程高性能的 Rust 随机数/随机过程模拟库，并利用 [PyO3](https://github.com/PyO3/pyo3) 提供 Python 封装。 Julia 版本正在同步开发中，可见 [DiffusionX.jl](https://github.com/tangxiangong/DiffusionX.jl)。

[![文档](https://img.shields.io/badge/文档-最新-blue.svg)](https://docs.rs/diffusionx/0.1.0/diffusionx/)

## 使用示例
### Python

```python
from diffusionx.simulation import Bm

# 布朗运动模拟
bm = Bm() 
traj = bm(10)
times, positions = traj.simulate(step_size=0.01)  # 模拟布朗运动轨迹，返回 ndarray 数组

# 蒙特卡罗模拟布朗运动的统计量
raw_moment = traj.raw_moment(order=1, particles=1000)  # 一阶原点矩
central_moment = traj.central_moment(order=2, particles=1000)  # 二阶中心矩

# 布朗运动首次通过时间
fpt = bm.fpt((-1, 1))
```

### Rust

### 开始使用
添加以下内容到您的 `Cargo.toml`:
```toml
[dependencies]
diffusionx = "*"
```
或者使用以下命令安装:
```bash
cargo add diffusionx
```

### 随机数生成

```rust
use diffusionx::random::{normal, uniform, exponential, poisson, stable};

// 正态分布
let normal_sample = normal::rand(0.0, 1.0)?; // 生成一个均值为 0.0，标准差为 1.0 的正态随机数
let normal_samples = normal::rands(2.0, 3.0, 1000)?; // 生成 1000 个均值为 2.0，标准差为 3.0 的正态随机数
let std_normal_sample = normal::standard_rand(); // 生成一个标准正态随机数 (均值 0，标准差 1)
let std_normal_samples = normal::standard_rands(1000);  // 生成 1000 个标准正态随机数

// 均匀分布
let uniform_sample = uniform::range_rand(0..10)?; // 生成一个 [0, 10) 范围内的均匀随机数
let uniform_samples = uniform::range_rands(0..10, 1000)?; // 生成 1000 个 [0, 10) 范围内的均匀随机数
let uniform_incl_sample = uniform::inclusive_range_rand(0..=10)?; // 生成一个 [0, 10] 范围内的均匀随机数
let uniform_incl_samples = uniform::inclusive_range_rands(0..=10, 1000)?; // 生成 1000 个 [0, 10] 范围内的均匀随机数
let std_uniform_sample = uniform::standard_rand(); // 生成一个 [0, 1) 范围内的均匀随机数
let std_uniform_samples = uniform::standard_rands(1000); // 生成 1000 个 [0, 1) 范围内的均匀随机数
let bool_sample = uniform::bool_rand(0.7)?; // 生成一个概率为 0.7 的布尔随机数
let bool_samples = uniform::bool_rands(0.7, 1000)?; // 生成 1000 个概率为 0.7 的布尔随机数

// 指数分布
let exp_sample = exponential::rand(1.0)?; // 生成一个速率为 1.0 的指数随机数
let exp_samples = exponential::rands(1.0, 1000)?; // 生成 1000 个速率为 1.0 的指数随机数

// 泊松分布
let poisson_sample = poisson::rand(5.0)?; // 生成一个均值为 5.0 的泊松随机数
let poisson_samples = poisson::rands(5.0, 1000)?; // 生成 1000 个均值为 5.0 的泊松随机数

// α-稳定分布
// 标准 α-稳定分布 (σ=1, μ=0)
let stable_sample = stable::standard_rand(1.5, 0.5)?; // 生成一个 α=1.5, β=0.5 的标准稳定随机数
let stable_samples = stable::standard_rands(1.5, 0.5, 1000)?; // 生成 1000 个标准稳定随机数

// 一般 α-稳定分布
let stable_sample = stable::rand(1.5, 0.5, 1.0, 0.0)?; // 生成一个 α=1.5, β=0.5, σ=1.0, μ=0.0 的稳定随机数
let stable_samples = stable::rands(1.5, 0.5, 1.0, 0.0, 1000)?; // 生成 1000 个稳定随机数

// α-稳定分布的特殊情况
let skew_sample = stable::skew_rand(1.5)?; // 生成一个 α=1.5 的完全倾斜的稳定随机数
let skew_samples = stable::skew_rands(1.5, 1000)?; // 生成 1000 个完全倾斜的稳定随机数
let sym_sample = stable::sym_standard_rand(1.5)?; // 生成一个 α=1.5 的对称稳定随机数
let sym_samples = stable::sym_standard_rands(1.5, 1000)?; // 生成 1000 个对称稳定随机数

// 稳定分布的面向对象接口
let stable = stable::Stable::new(1.5, 0.5, 1.0, 0.0)?; // 创建一个稳定分布对象
let samples = stable.samples(1000)?; // 生成 1000 个样本

let std_stable = stable::StandardStable::new(1.5, 0.5)?; // 创建一个标准稳定分布对象
let samples = std_stable.samples(1000)?; // 生成 1000 个样本
```

### 随机过程模拟

```rust
use diffusionx::simulation::{prelude::*, Bm};

// 布朗运动模拟
let bm = Bm::default();  // 创建标准布朗运动对象
let traj = bm.duration(1.0)?;  // 创建持续时间为 1.0 的轨迹
let (times, positions) = traj.simulate(0.01)?;  // 模拟布朗运动轨迹，时间步长为 0.01

// 布朗运动统计量蒙特卡罗模拟
let mean = traj.raw_moment(1, 1000, 0.01)?;  // 一阶原点矩，1000 个粒子，时间步长为 0.01
let msd = traj.central_moment(2, 1000, 0.01)?;  // 二阶中心矩，1000 个粒子，时间步长为 0.01

// 布朗运动首次通过时间
let max_duration = 1000; // 如果超过此时间，模拟将终止并返回 None
let fpt = bm.fpt(0.01, (-1.0, 1.0), max_duration)?; 
// 或者
let fpt = FirstPassageTime::new(&bm, (-1.0, 1.0))?;
let fpt_result = fpt.simulate(max_duration, 0.01)?;
```
## 可扩展性

DiffusionX 采用 trait 系统设计，具有高度的可扩展性：

### 核心 Trait

- `ContinuousProcess`: 连续随机过程的基本特征 trait
- `PointProcess`: 点过程的基本特征 trait
- `Moment`: 随机过程的统计量计算 trait，包括原点矩和中心矩

### 功能扩展

1. 添加新的连续随机过程：
   ```rust
   #[derive(Clone)]
   struct MyProcess {
       // 您的参数
   }
   
   impl ContinuousProcess for MyProcess {
       fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<(Vec<f64>, Vec<f64>)> {
           // 实现您的模拟逻辑
           todo!()
       }
   }
   ```

2. 自动获得功能：
   - 实现 `ContinuousProcess` trait 后自动获得 `ContinuousTrajectoryTrait` 功能
   - 通过 `ContinuousTrajectory` 获得 `Moment` 功能
   - 支持矩统计量计算

代码示例：
```rust
let myprocess = MyProcess::default();
let traj = myprocess.duration(10)?;
let (times, positions) = traj.simulate(0.01)?;
let mean = traj.raw_moment(1, 1000, 0.01)?;
let msd = traj.central_moment(2, 1000, 0.01)?;
```

3. 并行计算支持：
   - 矩计算自动支持并行计算
   - 统计量计算默认使用并行策略


## 进展
### 随机数生成

- [x] 正态分布
- [x] 均匀分布
- [x] 指数分布
- [x] 泊松分布
- [x] alpha 稳定分布

### 随机过程

- [x] 布朗运动
- [x] alpha 稳定 Levy 过程
- [x] 从属过程
- [x] 逆从属过程
- [x] 分数布朗运动
- [x] 泊松过程
- [ ] 复合泊松过程
- [x] Langevin 方程
- [x] 广义 Langevin 方程
- [x] 从属 Langevin 方程

### 泛函分布

- [x] 首次通过时间
- [x] 停留时间

## Benchmark

### 测试结果

生成长度为 `10_000_000` 的随机数组

|                          | 标准正态分布 | `[0, 1]` 均匀分布 |  稳定分布  |
| :----------------------: | :----------: | :---------------: | :--------: |
|  DiffusionX (Rust 版本)  |  17.576 ms   |     15.131 ms     | 133.85 ms  |
| DiffusionX (Python 版本) |   41.2 ms    |     34.3 ms     |  293 ms  |
|          Julia           |  27.671 ms   |     12.755 ms      | 570.260 ms |
|      NumPy / SciPy       |    199 ms    |      66.6 ms      |   1.67 s   |
|          Numba           |      -       |         -         |   1.15 s   |

### 测试环境

#### 硬件配置
- 设备型号：MacBook Air 13-inch (2024)
- 处理器：Apple M3 芯片
- 内存：16GB

#### 软件环境
- 操作系统：macOS Sequoia 15.3
- Rust：1.85.0
- Python：3.12
- Julia：1.11
- NumPy：2
- SciPy：1.15.1

## 技术栈 & 特性

- 🦀 Rust 2024 Edition
- 🔄 PyO3：Rust/Python 绑定
- 🔢 NumPy：零开销数组转换
- 🚀 高性能 
- 🔄 零开销 NumPy 兼容：所有随机数生成函数直接返回 NumPy 数组，无需额外转换

## 许可证

本项目采用双许可证模式：

* [MIT 许可证](https://opensource.org/licenses/MIT)
* [Apache 许可证 2.0 版本](https://www.apache.org/licenses/LICENSE-2.0)

您可以选择使用其中任一许可证。
