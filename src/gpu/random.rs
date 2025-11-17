//! GPU 随机数生成器
//!
//! 为所有 CPU 随机数分布提供 GPU 加速版本

use crate::XResult;

#[cfg(feature = "cuda")]
use super::cuda::{CudaBackend, KernelManager};

/// GPU 随机数生成器 trait
pub trait GpuRandom {
    /// 在 GPU 上生成随机数
    fn generate_gpu(&self, n: usize, backend: super::GpuBackend) -> XResult<Vec<f64>>;
}

/// GPU 均匀分布
pub struct GpuUniform {
    pub low: f64,
    pub high: f64,
}

impl GpuUniform {
    pub fn new(low: f64, high: f64) -> Self {
        Self { low, high }
    }
}

impl GpuRandom for GpuUniform {
    fn generate_gpu(&self, _n: usize, backend: super::GpuBackend) -> XResult<Vec<f64>> {
        match backend {
            #[cfg(feature = "cuda")]
            super::GpuBackend::Cuda => {
                use cudarc::curand::CurandGenerator;
                let cuda = CudaBackend::new(0)?;
                let device = cuda.device();

                let mut generator = CurandGenerator::new(12345, device.clone()).map_err(|e| {
                    crate::XError::GpuError(format!("Failed to create curand: {}", e))
                })?;

                let mut randoms = device
                    .alloc_zeros::<f32>(n)
                    .map_err(|e| crate::XError::GpuError(format!("Failed to allocate: {}", e)))?;

                generator
                    .fill_with_uniform(&mut randoms)
                    .map_err(|e| crate::XError::GpuError(format!("Failed to generate: {}", e)))?;

                let host_data = device
                    .dtoh_sync_copy(&randoms)
                    .map_err(|e| crate::XError::GpuError(format!("Failed to copy: {}", e)))?;

                // 缩放到 [low, high]
                let range = self.high - self.low;
                Ok(host_data
                    .iter()
                    .map(|&x| self.low + x as f64 * range)
                    .collect())
            }
            _ => Err(crate::XError::GpuError("Backend not supported".to_string())),
        }
    }
}

/// GPU 正态分布
pub struct GpuNormal {
    pub mean: f64,
    pub std_dev: f64,
}

impl GpuNormal {
    pub fn new(mean: f64, std_dev: f64) -> Self {
        Self { mean, std_dev }
    }
}

impl GpuRandom for GpuNormal {
    fn generate_gpu(&self, _n: usize, backend: super::GpuBackend) -> XResult<Vec<f64>> {
        match backend {
            #[cfg(feature = "cuda")]
            super::GpuBackend::Cuda => {
                use cudarc::curand::CurandGenerator;
                let cuda = CudaBackend::new(0)?;
                let device = cuda.device();

                let mut generator = CurandGenerator::new(12345, device.clone()).map_err(|e| {
                    crate::XError::GpuError(format!("Failed to create curand: {}", e))
                })?;

                let mut randoms = device
                    .alloc_zeros::<f32>(n)
                    .map_err(|e| crate::XError::GpuError(format!("Failed to allocate: {}", e)))?;

                generator
                    .fill_with_normal(&mut randoms, self.mean as f32, self.std_dev as f32)
                    .map_err(|e| crate::XError::GpuError(format!("Failed to generate: {}", e)))?;

                let host_data = device
                    .dtoh_sync_copy(&randoms)
                    .map_err(|e| crate::XError::GpuError(format!("Failed to copy: {}", e)))?;

                Ok(host_data.iter().map(|&x| x as f64).collect())
            }
            _ => Err(crate::XError::GpuError("Backend not supported".to_string())),
        }
    }
}

/// GPU 指数分布
pub struct GpuExponential {
    pub lambda: f64,
}

impl GpuExponential {
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl GpuRandom for GpuExponential {
    fn generate_gpu(&self, n: usize, backend: super::GpuBackend) -> XResult<Vec<f64>> {
        // 使用均匀分布通过逆变换采样
        let uniform = GpuUniform::new(0.0, 1.0);
        let u = uniform.generate_gpu(n, backend)?;
        Ok(u.iter().map(|&x| -(1.0 - x).ln() / self.lambda).collect())
    }
}

/// GPU Gamma 分布
pub struct GpuGamma {
    pub shape: f64,
    pub scale: f64,
}

impl GpuGamma {
    pub fn new(shape: f64, scale: f64) -> Self {
        Self { shape, scale }
    }
}

impl GpuRandom for GpuGamma {
    fn generate_gpu(&self, n: usize, _backend: super::GpuBackend) -> XResult<Vec<f64>> {
        // 简化实现：使用 CPU 并行
        use rand_distr::{Distribution, Gamma};
        use rayon::prelude::*;

        let gamma = Gamma::new(self.shape, self.scale)
            .map_err(|e| crate::XError::GpuError(format!("Invalid gamma params: {}", e)))?;

        let samples: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng();
                gamma.sample(&mut rng)
            })
            .collect();

        Ok(samples)
    }
}

/// GPU Poisson 分布
pub struct GpuPoisson {
    pub lambda: f64,
}

impl GpuPoisson {
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl GpuRandom for GpuPoisson {
    fn generate_gpu(&self, _n: usize, backend: super::GpuBackend) -> XResult<Vec<f64>> {
        match backend {
            #[cfg(feature = "cuda")]
            super::GpuBackend::Cuda => {
                use cudarc::curand::CurandGenerator;
                let cuda = CudaBackend::new(0)?;
                let device = cuda.device();

                let mut generator = CurandGenerator::new(12345, device.clone()).map_err(|e| {
                    crate::XError::GpuError(format!("Failed to create curand: {}", e))
                })?;

                let mut randoms = device
                    .alloc_zeros::<u32>(n)
                    .map_err(|e| crate::XError::GpuError(format!("Failed to allocate: {}", e)))?;

                generator
                    .fill_with_poisson(&mut randoms, self.lambda)
                    .map_err(|e| crate::XError::GpuError(format!("Failed to generate: {}", e)))?;

                let host_data = device
                    .dtoh_sync_copy(&randoms)
                    .map_err(|e| crate::XError::GpuError(format!("Failed to copy: {}", e)))?;

                Ok(host_data.iter().map(|&x| x as f64).collect())
            }
            _ => Err(crate::XError::GpuError("Backend not supported".to_string())),
        }
    }
}

/// GPU Stable 分布
pub struct GpuStable {
    pub alpha: f64,
    pub beta: f64,
}

impl GpuStable {
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }
}

impl GpuRandom for GpuStable {
    fn generate_gpu(&self, n: usize, backend: super::GpuBackend) -> XResult<Vec<f64>> {
        match backend {
            #[cfg(feature = "cuda")]
            super::GpuBackend::Cuda => {
                use crate::gpu::cuda::{CudaBackend, KernelManager};

                let cuda = CudaBackend::new(0)?;
                let device = cuda.device();

                let mut kernel_manager = KernelManager::new(device.clone());

                let samples_f32 = kernel_manager.generate_stable_f32(
                    n,
                    self.alpha as f32,
                    self.beta as f32,
                    12345u64,
                )?;

                Ok(samples_f32.iter().map(|&x| x as f64).collect())
            }
            #[cfg(feature = "metal")]
            super::GpuBackend::Metal => {
                use crate::gpu::metal::MetalBackend;

                let metal = MetalBackend::new()?;

                let samples_f32 =
                    metal.generate_stable(n, self.alpha as f32, self.beta as f32, 12345u32)?;

                Ok(samples_f32.iter().map(|&x| x as f64).collect())
            }
            _ => {
                // 回落到 CPU 并行实现
                use rayon::prelude::*;
                use std::f64::consts::PI;

                let samples: Vec<f64> = (0..n)
                    .into_par_iter()
                    .map(|_| {
                        let mut rng = rand::rng();
                        use rand::Rng;

                        let u: f64 = rng.random::<f64>() * PI - PI / 2.0;
                        let w: f64 = -rng.random::<f64>().ln();

                        if (self.alpha - 1.0).abs() < 1e-10 {
                            let xi = PI / 2.0 + self.beta * u;
                            (2.0 / PI) * (xi * u.tan() - self.beta * w.ln())
                        } else {
                            let zeta = -self.beta * (PI * self.alpha / 2.0).tan();
                            let xi = (1.0 + zeta * zeta).powf(1.0 / (2.0 * self.alpha)).atan()
                                / self.alpha;

                            let part1 = (self.alpha * (u + xi)).sin();
                            let part2 = u.cos().powf(1.0 / self.alpha);
                            let part3 = ((1.0 - self.alpha) * (u + xi)).cos() / w;
                            let part3 = part3.powf((1.0 - self.alpha) / self.alpha);

                            part1 / part2 * part3
                        }
                    })
                    .collect();

                Ok(samples)
            }
        }
    }
}
