#include <metal_stdlib>
using namespace metal;

/**
 * Metal Shader for Geometric Brownian Motion (geometric_bm.rs)
 *
 * Corresponds to: src/simulation/continuous/geometric_bm.rs
 *
 * SDE: dS_t = μ S_t dt + σ S_t dW_t
 */

/**
 * Simulate Geometric Brownian Motion trajectories (single precision)
 *
 * @param random_normals Pre-generated normal random numbers [num_particles * num_steps]
 * @param positions Output array [num_particles * (num_steps + 1)]
 * @param start_position Initial position S_0
 * @param mu Drift coefficient (μ)
 * @param sigma Volatility (σ)
 * @param time_step Time step (dt)
 * @param num_steps Number of time steps
 * @param num_particles Number of particles to simulate
 * @param particle_idx Thread index (particle ID)
 */
kernel void simulate_geometric_bm(
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
    int offset = particle_idx * (num_steps + 1);
    int random_offset = particle_idx * num_steps;

    // Store initial position
    positions[offset] = position;

    float sqrt_dt = sqrt(time_step);

    // Simulate trajectory
    for (int step = 0; step < num_steps; step++) {
        float random_normal = random_normals[random_offset + step];

        // GBM exact solution: S(t+dt) = S(t) * exp((μ - σ²/2)dt + σ√dt Z)
        float drift = (mu - 0.5f * sigma * sigma) * time_step;
        float diffusion = sigma * sqrt_dt * random_normal;
        position *= exp(drift + diffusion);

        positions[offset + step + 1] = position;
    }
}
