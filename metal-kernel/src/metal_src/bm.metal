#include <metal_stdlib>
using namespace metal;

/**
 * Metal Shader for Brownian Motion (bm.rs)
 *
 * Corresponds to: src/simulation/continuous/bm.rs
 */

/**
 * Simulate Brownian motion trajectories (single precision)
 *
 * @param random_normals Pre-generated normal random numbers [num_particles * num_steps]
 * @param positions Output array [num_particles * (num_steps + 1)]
 * @param start_position Initial position for all particles
 * @param diffusion_coefficient Diffusion coefficient (D)
 * @param time_step Time step (dt)
 * @param num_steps Number of time steps
 * @param num_particles Number of particles to simulate
 * @param particle_idx Thread index (particle ID)
 */
kernel void simulate_bm(
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
    int offset = particle_idx * (num_steps + 1);
    int random_offset = particle_idx * num_steps;

    // Store initial position
    positions[offset] = position;

    // Precompute noise scaling: sqrt(2 * D * dt)
    float noise_scale = sqrt(2.0f * diffusion_coefficient * time_step);

    // Simulate trajectory
    for (int step = 0; step < num_steps; step++) {
        float random_normal = random_normals[random_offset + step];
        position += noise_scale * random_normal;
        positions[offset + step + 1] = position;
    }
}
