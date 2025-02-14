# DiffusionX

DiffusionX 是一个高性能的 Rust 反常扩散模拟库，并提供 Python 接口。

## 技术栈

- 🦀 Rust 2024 Edition
- 🔄 PyO3：Rust/Python 绑定
- 🔢 NumPy：零开销数组转换

## 特性

- 🚀 高性能 
- 🔄 零开销 NumPy 兼容：所有随机数生成函数直接返回 NumPy 数组，无需额外转换
- 🎯 类型安全：支持浮点数和整数类型的随机数生成
- 🛡️ 内存安全：基于 Rust 实现，保证内存安全和线程安全

## 使用示例

### Rust

```rust
use diffusionx::random;

// 生成标准正态分布随机数
let value = random::normal::standard_rand();
let values = random::normal::standard_rands(1000);

// 生成均匀分布随机数
let value = random::uniform::standard_rand();
let values = random::uniform::standard_rands(1000);
let values = random::uniform::range_rands(0..10, 1000);
let values = random::uniform::inclusive_range_rands(0..=10, 1000);

// 生成指数分布随机数
let value = random::exponential::rand(1.0);
let values = random::exponential::rands(1.0, 1000);
```

### Python

```python
from diffusionx import random
from diffusionx.types import DType

# 生成正态分布随机数
value = random.randn()  # 生成一个标准正态分布随机数
values = random.randn(1000, mu=0.0, sigma=1.0)  # 生成1000个正态分布随机数

# 生成均匀分布随机数
value = random.uniform()  # 生成一个[0,1)均匀分布随机数
values = random.uniform(1000, low=0.0, high=1.0, dtype=DType.Float)  # 生成1000个浮点型均匀分布随机数
values_int = random.uniform(1000, low=0, high=100, dtype=DType.Int)  # 生成1000个整型均匀分布随机数

# 生成指数分布随机数
value = random.randexp()  # 生成一个参数为1的指数分布随机数
values = random.randexp(1000, scale=2.0)  # 生成1000个参数为2的指数分布随机数
```

所有返回多个随机数的函数都直接返回 NumPy 数组，可以无缝集成到现有的 NumPy 代码中：

```python
import numpy as np
from diffusionx import random

# DiffusionX 生成的数组可以直接用于 NumPy 运算
values = random.randn(1000)
mean = np.mean(values)  # 计算均值
std = np.std(values)   # 计算标准差

# 可以直接与 NumPy 数组进行运算
array1 = random.uniform(1000)
array2 = np.array([1, 2, 3, 4, 5])
result = array1[:5] + array2  # 数组相加
```

## 性能基准测试

生成长度为 `100_000_000` 的随机数组
|  | 标准正态分布 | `[0, 1]` 均匀分布 |
| :---: | :---: | :---: |
| [DiffusionX (Rust)](./diffusionx/)  | 273.85 ms | 245.78 ms |
| [DiffusionX (Python)](./py-diffusionx/) | 310 ms | 252 ms |
| [Julia](https://julialang.org/) | 581.61 ms | 371.37 ms |
| [NumPy](https://numpy.org/) | 3.28 s | 1.15 s |
| [Octave](https://octave.org/) | 1.31 s | 1.01 s |
| [Baltamatica](https://www.baltamatica.com/) | 5.47 s | 1.09 s |

## 许可证

本项目采用双许可证模式：

* [MIT 许可证](https://opensource.org/licenses/MIT)
* [Apache 许可证 2.0 版本](https://www.apache.org/licenses/LICENSE-2.0)

您可以选择使用其中任一许可证。
