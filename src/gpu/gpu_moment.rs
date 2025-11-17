//! GPU 加速的矩计算 Trait
//!
//! 为所有 Trajectory 类型（连续、离散、点过程）自动添加 `_gpu` 后缀的 GPU 计算方法。

use super::{GpuBackend, GpuSimulator};
use crate::{
    XResult,
    simulation::prelude::{
        ContinuousProcess, ContinuousTrajectory, DiscreteProcess, DiscreteTrajectory, PointProcess,
        PointTrajectory,
    },
};

/// 为所有 Trajectory 类型提供 GPU 矩计算方法的 Trait
pub trait GpuMoment {
    /// 在 GPU 上计算原始矩
    fn raw_moment_gpu(
        &self,
        order: i32,
        particles: usize,
        time_step: f64,
        backend: GpuBackend,
    ) -> XResult<f64>;

    /// 在 GPU 上计算中心矩
    fn central_moment_gpu(
        &self,
        order: i32,
        particles: usize,
        time_step: f64,
        backend: GpuBackend,
    ) -> XResult<f64>;
}

// 1. 为连续过程实现
impl<SP: ContinuousProcess + Clone> GpuMoment for ContinuousTrajectory<SP> {
    fn raw_moment_gpu(
        &self,
        order: i32,
        particles: usize,
        time_step: f64,
        backend: GpuBackend,
    ) -> XResult<f64> {
        let simulator = GpuSimulator::new(backend)?;
        simulator.raw_moment(&self.sp, order, particles, self.duration, time_step)
    }

    fn central_moment_gpu(
        &self,
        order: i32,
        particles: usize,
        time_step: f64,
        backend: GpuBackend,
    ) -> XResult<f64> {
        let simulator = GpuSimulator::new(backend)?;
        simulator.central_moment(&self.sp, order, particles, self.duration, time_step)
    }
}

// 2. 为离散过程实现
impl<SP: DiscreteProcess + Clone> GpuMoment for DiscreteTrajectory<SP> {
    fn raw_moment_gpu(
        &self,
        order: i32,
        particles: usize,
        _time_step: f64,
        backend: GpuBackend,
    ) -> XResult<f64> {
        let simulator = GpuSimulator::new(backend)?;
        simulator.raw_moment_discrete(&self.sp, order, particles, self.num_step)
    }

    fn central_moment_gpu(
        &self,
        order: i32,
        particles: usize,
        _time_step: f64,
        backend: GpuBackend,
    ) -> XResult<f64> {
        let simulator = GpuSimulator::new(backend)?;
        simulator.central_moment_discrete(&self.sp, order, particles, self.num_step)
    }
}

// 3. 为点过程实现
impl<SP: PointProcess> GpuMoment for PointTrajectory<SP> {
    fn raw_moment_gpu(
        &self,
        order: i32,
        particles: usize,
        _time_step: f64,
        backend: GpuBackend,
    ) -> XResult<f64> {
        let simulator = GpuSimulator::new(backend)?;
        let duration = self.duration.ok_or_else(|| {
            crate::SimulationError::InvalidParameters(
                "Duration must be set for point process moment calculation".to_string(),
            )
        })?;
        // 注意：点过程使用 duration，sp 是 Arc<SP>，需要解引用
        simulator.raw_moment_point(self.sp.as_ref(), order, particles, duration)
    }

    fn central_moment_gpu(
        &self,
        order: i32,
        particles: usize,
        _time_step: f64,
        backend: GpuBackend,
    ) -> XResult<f64> {
        let simulator = GpuSimulator::new(backend)?;
        let duration = self.duration.ok_or_else(|| {
            crate::SimulationError::InvalidParameters(
                "Duration must be set for point process moment calculation".to_string(),
            )
        })?;
        simulator.central_moment_point(self.sp.as_ref(), order, particles, duration)
    }
}
