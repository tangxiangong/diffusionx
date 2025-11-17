//! GPU 加速的 Trait 扩展
//!
//! 为所有 `ContinuousProcess` 类型自动添加 `_gpu` 后缀的 GPU 计算方法。

use super::{GpuBackend, GpuSimulator};
use crate::{XResult, simulation::prelude::ContinuousProcess};

/// 为 `ContinuousProcess` 提供 GPU 加速方法的 Trait
pub trait GpuProcess: ContinuousProcess {
    /// 在 GPU 上计算原始矩
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use diffusionx::simulation::continuous::Bm;
    /// use diffusionx::gpu::{GpuProcess, GpuBackend};
    ///
    /// let bm = Bm::default();
    /// let mean = bm.raw_moment_gpu(1, 10000, 0.01, GpuBackend::Auto)?;
    /// ```
    fn raw_moment_gpu(
        &self,
        order: i32,
        particles: usize,
        time_step: f64,
        backend: GpuBackend,
    ) -> XResult<f64> {
        let simulator = GpuSimulator::new(backend)?;
        simulator.raw_moment(self, order, particles, time_step)
    }

    /// 在 GPU 上计算中心矩
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let variance = bm.central_moment_gpu(2, 10000, 0.01, GpuBackend::Auto)?;
    /// ```
    fn central_moment_gpu(
        &self,
        order: i32,
        particles: usize,
        time_step: f64,
        backend: GpuBackend,
    ) -> XResult<f64> {
        let simulator = GpuSimulator::new(backend)?;
        simulator.central_moment(self, order, particles, time_step)
    }
}

// 为所有实现了 `ContinuousProcess` 的类型自动实现 `GpuProcess`
impl<T: ContinuousProcess> GpuProcess for T {}
