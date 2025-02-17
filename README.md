# DiffusionX
> [!NOTE]
> Development is in progress. DiffusionX 是一个多线程高性能的 Rust 随机数/随机过程模拟库，并利用 [PyO3](https://github.com/PyO3/pyo3) 提供 Python 封装。

## 使用示例
### Python

```python
from diffusionx.simulation import Bm

# 布朗运动模拟
bm = Bm(10) 
times, positions = bm.simulate(step_size=0.01)  # 模拟布朗运动轨迹，返回 ndarray 数组

# 蒙特卡罗模拟布朗运动的统计量
raw_moment = bm.raw_moment(order=1, particles=1000)  # 一阶原点矩
central_moment = bm.central_moment(order=2, particles=1000)  # 二阶中心矩

# 布朗运动首次穿越时间
fpt = bm.fpt((-1, 1))
```

### Rust

```rust
use diffusionx::simulation::{Bm, Simulation, Functional};

// 布朗运动模拟
let bm = Bm::new(0.0, 1.0, 1.0)?;  // 创建布朗运动对象：起始位置为0，扩散系数为1，持续时间为1
let time_step = 0.01;  // 时间步长
let (times, positions) = bm.simulate(time_step)?;  // 模拟布朗运动轨迹

// 蒙特卡罗模拟布朗运动的统计量
let mean = bm.mean(time_step, 1000)?;  // 均值  bm.raw_moment(time_step, 1, 1000)?;
let msd = bm.msd(time_step, 1000)?;  // 均方位移  bm.central_moment(time_step, 2, 1000)?;

// 布朗运动首次穿越时间
let fpt = bm.fpt(time_step, (-1.0, 1.0))?; 
```

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
