#include <metal_stdlib>
using namespace metal;

/**
 * Metal Shader for Fractional Brownian Motion (fbm.rs)
 * Corresponds to: src/simulation/continuous/fbm.rs
 */

kernel void simulate_fbm(
    device const float* random_normals [[buffer(0)]],
    device float* positions [[buffer(1)]],
    device const float* covariance_matrix [[buffer(2)]],
    constant float& hurst [[buffer(3)]],
    constant float& time_step [[buffer(4)]],
    constant int& num_steps [[buffer(5)]],
    constant int& num_particles [[buffer(6)]],
    uint particle_idx [[thread_position_in_grid]]
) {
    if (particle_idx >= uint(num_particles)) return;

    int offset = particle_idx * (num_steps + 1);
    int random_offset = particle_idx * num_steps;

    positions[offset] = 0.0f;

    float position = 0.0f;
    for (int step = 0; step < num_steps; step++) {
        float increment = 0.0f;
        for (int j = 0; j <= step; j++) {
            increment += covariance_matrix[step * num_steps + j] * 
                        random_normals[random_offset + j];
        }
        position += increment;
        positions[offset + step + 1] = position;
    }
}
