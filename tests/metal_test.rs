use diffusionx::XResult;
use diffusionx::gpu::{GpuBackend, GpuSimulator};
use diffusionx::simulation::continuous::Bm;

fn main() -> XResult<()> {
    println!("=== Metal GPU 测试 ===\n");

    // 1. 检查 Metal 是否可用
    println!("1. 检查 Metal 可用性...");
    if !diffusionx::gpu::metal::is_available() {
        println!("❌ Metal 不可用！");
        return Ok(());
    }
    println!("✅ Metal 可用");

    // 2. 获取设备数量
    match diffusionx::gpu::metal::device_count() {
        Ok(count) => println!("   设备数量: {}", count),
        Err(e) => {
            println!("❌ 无法获取设备数量: {}", e);
            return Ok(());
        }
    }

    // 3. 创建 Metal 后端并显示设备信息
    println!("\n2. 创建 Metal 后端...");
    match diffusionx::gpu::metal::MetalBackend::new() {
        Ok(backend) => {
            let info = backend.device_info();
            println!("✅ Metal 后端创建成功");
            println!("   设备名称: {}", info.name);
            println!("   低功耗模式: {}", info.is_low_power);
            println!(
                "   最大工作集大小: {} MB",
                info.recommended_max_working_set_size / 1024 / 1024
            );
            println!(
                "   最大线程组大小: {} x {} x {}",
                info.max_threads_per_threadgroup.width,
                info.max_threads_per_threadgroup.height,
                info.max_threads_per_threadgroup.depth
            );
        }
        Err(e) => {
            println!("❌ 创建 Metal 后端失败: {}", e);
            return Ok(());
        }
    }

    // 4. 创建 GPU 模拟器
    println!("\n3. 创建 GPU 模拟器...");
    let simulator = match GpuSimulator::new(GpuBackend::Metal) {
        Ok(sim) => {
            println!("✅ GPU 模拟器创建成功");
            println!("   {}", sim.info());
            sim
        }
        Err(e) => {
            println!("❌ 创建 GPU 模拟器失败: {}", e);
            return Ok(());
        }
    };

    // 5. 测试布朗运动模拟
    println!("\n4. 测试布朗运动模拟...");
    let bm = Bm::default();
    let duration: f64 = 1.0;
    let time_step: f64 = 0.01;
    let num_particles = 100;

    println!("   参数:");
    println!("   - 粒子数量: {}", num_particles);
    println!("   - 模拟时长: {} 秒", duration);
    println!("   - 时间步长: {}", time_step);
    println!("   - 预期步数: {}", (duration / time_step).ceil() as usize);

    let start = std::time::Instant::now();
    match simulator.simulate_bm_metal(&bm, duration, time_step, num_particles) {
        Ok(trajectories) => {
            let elapsed = start.elapsed();
            println!("✅ 模拟成功完成！");
            println!("   耗时: {:?}", elapsed);
            println!("   轨迹数量: {}", trajectories.len());

            if !trajectories.is_empty() {
                let (times, positions) = &trajectories[0];
                println!("   第一条轨迹:");
                println!("     - 时间点数量: {}", times.len());
                println!("     - 位置点数量: {}", positions.len());
                println!("     - 初始位置: {:.6}", positions[0]);
                println!("     - 最终位置: {:.6}", positions[positions.len() - 1]);

                // 计算简单统计
                let mean = positions.iter().sum::<f64>() / positions.len() as f64;
                let variance = positions.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / positions.len() as f64;
                println!("     - 均值: {:.6}", mean);
                println!("     - 方差: {:.6}", variance);
            }

            // 显示一些轨迹的终点统计
            let final_positions: Vec<f64> = trajectories
                .iter()
                .map(|(_, pos)| *pos.last().unwrap())
                .collect();
            let mean_final = final_positions.iter().sum::<f64>() / final_positions.len() as f64;
            println!("\n   所有轨迹终点统计:");
            println!("     - 均值: {:.6}", mean_final);
            println!(
                "     - 最小值: {:.6}",
                final_positions
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min)
            );
            println!(
                "     - 最大值: {:.6}",
                final_positions
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
            );
        }
        Err(e) => {
            println!("❌ 模拟失败: {}", e);
            return Err(e);
        }
    }

    // 6. 性能测试
    println!("\n5. 性能测试（更多粒子）...");
    let num_particles_large = 1000;
    println!("   粒子数量: {}", num_particles_large);

    let start = std::time::Instant::now();
    match simulator.simulate_bm_metal(&bm, duration, time_step, num_particles_large) {
        Ok(trajectories) => {
            let elapsed = start.elapsed();
            println!("✅ 大规模模拟完成！");
            println!("   轨迹数量: {}", trajectories.len());
            println!("   总耗时: {:?}", elapsed);
            println!(
                "   每个粒子平均耗时: {:?}",
                elapsed / num_particles_large as u32
            );
        }
        Err(e) => {
            println!("❌ 大规模模拟失败: {}", e);
        }
    }

    println!("\n=== 测试完成 ===");
    Ok(())
}
