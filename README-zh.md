# DiffusionX

[English](README.md) | 简体中文

> [!NOTE]
> 开发进行中。DiffusionX 是一个多线程高性能的 Rust 随机数/随机过程模拟库，并利用 [PyO3](https://github.com/PyO3/pyo3) 提供 Python 封装。 Julia 版本正在同步开发中，可见 [DiffusionX.jl](https://github.com/tangxiangong/DiffusionX.jl)。

[![文档](https://img.shields.io/badge/文档-最新-blue.svg)](https://tangxiangong.github.io/diffusionx/rust/diffusionx/index.html)

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

```rust
use diffusionx::simulation::{Bm, Simulation, Functional};

// 布朗运动模拟
let bm = Bm::default();  // 标准布朗运动
let time_step = 0.01;  // 时间步长
let duration = 1.0;  // 模拟时间
let (times, positions) = bm.simulate(duration, time_step)?;  // 模拟布朗运动轨迹

// 蒙特卡罗模拟布朗运动的统计量
let mean = bm.mean(duration, 1000, time_step)?;  // 均值  bm.raw_moment(duration, 1, 1000, time_step)?;
let msd = bm.msd(duration, 1000, time_step)?;  // 均方位移  bm.central_moment(duration, 2, 1000, time_step)?;

// 布朗运动首次通过时间
let max_duration = 1000; // 如果超过此时间，模拟将终止并返回 None
let fpt = bm.fpt((-1.0, 1.0), max_duration, time_step)?;  
// or
let fpt = FirstPassageTime::new(&bm, (-1.0, 1.0))?;
let fpt_result = fpt.simulate(max_duration, time_step)?;
```

## 可扩展性

DiffusionX 采用 trait 系统设计，具有高度的可扩展性：

### 核心 Trait

- `Stochastic`: 随机过程的基本特征 trait
- `Simulation`: 随机过程模拟的核心 trait，定义了模拟方法
- `Moment`: 随机过程的统计量计算 trait，包括原点矩和中心矩
- `Trajectory`: 随机过程轨迹处理 trait，提供轨迹相关功能

### 功能扩展

1. 添加新的随机过程：
   ```rust
   #[derive(Clone)]
   struct MyProcess {
       // 您的参数
   }
   
   impl Stochastic for MyProcess {}
   
   impl Simulation for MyProcess {
       fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<(Vec<f64>, Vec<f64>)> {
           // 实现您的模拟逻辑
       }
   }
   ```

2. 自动获得功能：
   - 实现 `Simulation` trait 后自动获得 `Trajectory` 和 `Moment` 功能
   - 可直接使用 `FirstPassageTime` 等功能性结构
   - 支持矩统计量计算
代码示例
```rust
let myprocess = MyProcess::default();
let traj = myprocess.duration(10)?;
let mean = traj.raw_moment(1, 1000, time_step)?;
let msd = traj.central_moment(2, 1000, time_step)?;
let fpt = FirstPassageTime::new(&myprocess, (-1.0, 1.0))?;
let fpt_result = fpt.simulate(max_duration, time_step)?;
let fpt_mean = fpt.raw_moment(1, 1000, time_step)?;
```

3. 并行计算支持：
   - 所有实现了 `Simulation` trait 的类型自动支持并行计算
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
- [ ] alpha 稳定 Levy 过程
- [ ] 分数布朗运动
- [ ] 泊松过程
- [ ] 复合泊松过程
- [ ] Langevin 方程


## Benchmark

### 测试结果

生成长度为 `10_000_000` 的随机数组

|                          | 标准正态分布 | `[0, 1]` 均匀分布 |  稳定分布  |
| :----------------------: | :----------: | :---------------: | :--------: |
|  DiffusionX (Rust 版本)  |  23.811 ms   |     20.450 ms     | 273.68 ms  |
| DiffusionX (Python 版本) |   24.1 ms    |     21.687 ms     |  277.6 ms  |
|          Julia           |  28.748 ms   |     9.748 ms      | 713.955 ms |
|      NumPy / SciPy       |    295 ms    |      81.2 ms      |   3.39 s   |
|          Numba           |      -       |         -         |   1.52 s   |

### 测试环境

#### 硬件配置
- 设备型号：MacBook Pro 13-inch (2020)
- 处理器：Intel Core i5-1038NG7 @ 2.0GHz (4核8线程)
- 内存：16GB LPDDR4X 3733MHz

#### 软件环境
- 操作系统：macOS Sequoia 15.3
- Rust：1.85.0-beta.7
- Python：3.12
- Julia：1.11
- NumPy：2.2.2
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
