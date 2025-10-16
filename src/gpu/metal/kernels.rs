//! Metal shader kernels for stochastic process simulation
//!
//! This module contains Metal Shading Language (MSL) kernels for GPU-accelerated
//! simulation of stochastic processes on Apple hardware.

/// Metal shader source code for all stochastic processes
pub const METAL_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Brownian Motion Kernels
// ============================================================================

/// Simulate Brownian motion (f32)
kernel void simulate_brownian_motion(
    device const float* random_normals [[buffer(0)]],
    device float* positions [[buffer(1)]],
    constant float& start_position [[buffer(2)]],
    constant float& diffusion_coefficient [[buffer(3)]],
    constant float& time_step [[buffer(4)]],
    constant int& num_steps [[buffer(5)]],
    constant int& num_particles [[buffer(6)]],
    uint particle_idx [[thread_position_in_grid]]
) {
    if (particle_idx >= uint(num_particles)) return;

    float position = start_position;
    int offset = particle_idx * num_steps;

    // Store initial position
    positions[offset] = position;

    // Simulate trajectory
    float noise_scale = sqrt(2.0f * diffusion_coefficient * time_step);
    for (int step = 0; step < num_steps; step++) {
        position += noise_scale * random_normals[offset + step];
        positions[offset + step + 1] = position;
    }
}

/// Simulate Brownian motion (f64 using double precision if available)
kernel void simulate_brownian_motion_f64(
    device const float* random_normals [[buffer(0)]],
    device float* positions [[buffer(1)]],
    constant float& start_position [[buffer(2)]],
    constant float& diffusion_coefficient [[buffer(3)]],
    constant float& time_step [[buffer(4)]],
    constant int& num_steps [[buffer(5)]],
    constant int& num_particles [[buffer(6)]],
    uint particle_idx [[thread_position_in_grid]]
) {
    // Note: Metal doesn't always support double precision
    // Using float for compatibility
    if (particle_idx >= uint(num_particles)) return;

    float position = start_position;
    int offset = particle_idx * num_steps;

    positions[offset] = position;

    float noise_scale = sqrt(2.0f * diffusion_coefficient * time_step);
    for (int step = 0; step < num_steps; step++) {
        position += noise_scale * random_normals[offset + step];
        positions[offset + step + 1] = position;
    }
}

// ============================================================================
// Ornstein-Uhlenbeck Process Kernels
// ============================================================================

/// Simulate Ornstein-Uhlenbeck process
kernel void simulate_ou_process(
    device const float* random_normals [[buffer(0)]],
    device float* positions [[buffer(1)]],
    constant float& start_position [[buffer(2)]],
    constant float& theta [[buffer(3)]],
    constant float& mu [[buffer(4)]],
    constant float& sigma [[buffer(5)]],
    constant float& time_step [[buffer(6)]],
    constant int& num_steps [[buffer(7)]],
    constant int& num_particles [[buffer(8)]],
    uint particle_idx [[thread_position_in_grid]]
) {
    if (particle_idx >= uint(num_particles)) return;

    float position = start_position;
    int offset = particle_idx * num_steps;

    positions[offset] = position;

    float sqrt_dt = sqrt(time_step);
    for (int step = 0; step < num_steps; step++) {
        float drift = theta * (mu - position) * time_step;
        float diffusion = sigma * sqrt_dt * random_normals[offset + step];
        position += drift + diffusion;
        positions[offset + step + 1] = position;
    }
}

// ============================================================================
// Geometric Brownian Motion Kernels
// ============================================================================

/// Simulate Geometric Brownian Motion
kernel void simulate_gbm(
    device const float* random_normals [[buffer(0)]],
    device float* positions [[buffer(1)]],
    constant float& start_position [[buffer(2)]],
    constant float& mu [[buffer(3)]],
    constant float& sigma [[buffer(4)]],
    constant float& time_step [[buffer(5)]],
    constant int& num_steps [[buffer(6)]],
    constant int& num_particles [[buffer(7)]],
    uint particle_idx [[thread_position_in_grid]]
) {
    if (particle_idx >= uint(num_particles)) return;

    float position = start_position;
    int offset = particle_idx * num_steps;

    positions[offset] = position;

    float sqrt_dt = sqrt(time_step);
    for (int step = 0; step < num_steps; step++) {
        float drift = (mu - 0.5f * sigma * sigma) * time_step;
        float diffusion = sigma * sqrt_dt * random_normals[offset + step];
        position *= exp(drift + diffusion);
        positions[offset + step + 1] = position;
    }
}

// ============================================================================
// Langevin Equation Kernels
// ============================================================================

/// Simulate Langevin equation
kernel void simulate_langevin(
    device const float* random_normals [[buffer(0)]],
    device float* positions [[buffer(1)]],
    device float* velocities [[buffer(2)]],
    constant float& start_position [[buffer(3)]],
    constant float& start_velocity [[buffer(4)]],
    constant float& gamma [[buffer(5)]],
    constant float& temperature [[buffer(6)]],
    constant float& mass [[buffer(7)]],
    constant float& time_step [[buffer(8)]],
    constant int& num_steps [[buffer(9)]],
    constant int& num_particles [[buffer(10)]],
    uint particle_idx [[thread_position_in_grid]]
) {
    if (particle_idx >= uint(num_particles)) return;

    float position = start_position;
    float velocity = start_velocity;
    int offset = particle_idx * num_steps;

    positions[offset] = position;
    velocities[offset] = velocity;

    float sqrt_dt = sqrt(time_step);
    float noise_scale = sqrt(2.0f * gamma * temperature / mass);

    for (int step = 0; step < num_steps; step++) {
        float dv = -gamma * velocity * time_step / mass +
                   noise_scale * random_normals[offset + step] * sqrt_dt;
        velocity += dv;
        position += velocity * time_step;

        positions[offset + step + 1] = position;
        velocities[offset + step + 1] = velocity;
    }
}

// ============================================================================
// Statistical Moment Computation Kernels
// ============================================================================

/// Compute raw moment of a given order
kernel void compute_raw_moment(
    device const float* positions [[buffer(0)]],
    device float* moments [[buffer(1)]],
    constant int& order [[buffer(2)]],
    constant int& num_particles [[buffer(3)]],
    constant int& num_steps [[buffer(4)]],
    uint step_idx [[thread_position_in_grid]]
) {
    if (step_idx >= uint(num_steps)) return;

    float sum = 0.0f;
    for (int i = 0; i < num_particles; i++) {
        float pos = positions[i * (num_steps + 1) + step_idx];
        float value = pos;
        for (int j = 1; j < order; j++) {
            value *= pos;
        }
        sum += value;
    }

    moments[step_idx] = sum / float(num_particles);
}

/// Compute central moment of a given order
kernel void compute_central_moment(
    device const float* positions [[buffer(0)]],
    device const float* means [[buffer(1)]],
    device float* moments [[buffer(2)]],
    constant int& order [[buffer(3)]],
    constant int& num_particles [[buffer(4)]],
    constant int& num_steps [[buffer(5)]],
    uint step_idx [[thread_position_in_grid]]
) {
    if (step_idx >= uint(num_steps)) return;

    float mean = means[step_idx];
    float sum = 0.0f;

    for (int i = 0; i < num_particles; i++) {
        float pos = positions[i * (num_steps + 1) + step_idx];
        float deviation = pos - mean;
        float value = deviation;
        for (int j = 1; j < order; j++) {
            value *= deviation;
        }
        sum += value;
    }

    moments[step_idx] = sum / float(num_particles);
}

/// Compute mean square displacement (optimized)
kernel void compute_msd(
    device const float* positions [[buffer(0)]],
    device float* msd [[buffer(1)]],
    constant float& start_position [[buffer(2)]],
    constant int& num_particles [[buffer(3)]],
    constant int& num_steps [[buffer(4)]],
    uint step_idx [[thread_position_in_grid]]
) {
    if (step_idx >= uint(num_steps)) return;

    float sum = 0.0f;
    for (int i = 0; i < num_particles; i++) {
        float pos = positions[i * (num_steps + 1) + step_idx];
        float displacement = pos - start_position;
        sum += displacement * displacement;
    }

    msd[step_idx] = sum / float(num_particles);
}

// ============================================================================
// Random Number Generation Utilities
// ============================================================================

/// Box-Muller transform to generate normal random numbers from uniform
kernel void box_muller_transform(
    device const float* uniform1 [[buffer(0)]],
    device const float* uniform2 [[buffer(1)]],
    device float* normal1 [[buffer(2)]],
    device float* normal2 [[buffer(3)]],
    constant int& n [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= uint(n)) return;

    float u1 = uniform1[idx];
    float u2 = uniform2[idx];

    // Avoid log(0)
    if (u1 < 1e-10f) u1 = 1e-10f;

    float r = sqrt(-2.0f * log(u1));
    float theta = 2.0f * M_PI_F * u2;

    normal1[idx] = r * cos(theta);
    normal2[idx] = r * sin(theta);
}

// ============================================================================
// First Passage Time Detection
// ============================================================================

/// Check if trajectory crosses boundary
kernel void check_boundary_crossing(
    device const float* positions [[buffer(0)]],
    device int* crossing_times [[buffer(1)]],
    constant float& lower_bound [[buffer(2)]],
    constant float& upper_bound [[buffer(3)]],
    constant int& num_steps [[buffer(4)]],
    uint particle_idx [[thread_position_in_grid]]
) {
    int offset = particle_idx * (num_steps + 1);
    crossing_times[particle_idx] = -1; // -1 means no crossing

    for (int step = 0; step <= num_steps; step++) {
        float pos = positions[offset + step];
        if (pos <= lower_bound || pos >= upper_bound) {
            crossing_times[particle_idx] = step;
            break;
        }
    }
}

// ============================================================================
// TAMSD (Time-Averaged Mean Square Displacement) Computation
// ============================================================================

/// Compute time-averaged MSD for a single trajectory
kernel void compute_single_tamsd(
    device const float* positions [[buffer(0)]],
    device float* tamsd [[buffer(1)]],
    constant float& delta [[buffer(2)]],
    constant int& num_steps [[buffer(3)]],
    constant int& delta_steps [[buffer(4)]],
    uint particle_idx [[thread_position_in_grid]]
) {
    int offset = particle_idx * (num_steps + 1);
    float sum = 0.0f;
    int count = 0;

    for (int t = 0; t <= num_steps - delta_steps; t++) {
        float pos_t = positions[offset + t];
        float pos_t_delta = positions[offset + t + delta_steps];
        float diff = pos_t_delta - pos_t;
        sum += diff * diff;
        count++;
    }

    tamsd[particle_idx] = (count > 0) ? (sum / float(count)) : 0.0f;
}

// ============================================================================
// Utility Kernels
// ============================================================================

/// Cumulative sum (prefix sum) - simplified version
kernel void cumsum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& initial_value [[buffer(2)]],
    constant int& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= uint(n)) return;

    // Note: This is a naive implementation
    // For better performance, use parallel prefix sum algorithm
    float sum = initial_value;
    for (int i = 0; i <= int(idx); i++) {
        sum += input[i];
    }
    output[idx] = sum;
}

/// Vector addition
kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant int& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= uint(n)) return;
    result[idx] = a[idx] + b[idx];
}

/// Scalar multiplication
kernel void scalar_multiply(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant int& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= uint(n)) return;
    output[idx] = input[idx] * scalar;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_shaders_not_empty() {
        assert!(!METAL_SHADERS.is_empty());
        assert!(METAL_SHADERS.contains("simulate_brownian_motion"));
        assert!(METAL_SHADERS.contains("simulate_ou_process"));
        assert!(METAL_SHADERS.contains("simulate_gbm"));
        assert!(METAL_SHADERS.contains("compute_raw_moment"));
    }
}
