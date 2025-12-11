/**
 * Metal Kernel for Brownian motion (bm.rs)
 *
 * Corresponds to: src/simulation/continuous/bm.rs and src/gpu/metal/bm.rs
 *
 * This file implements various Metal compute kernels for computing statistical properties
 * of Brownian motion processes through Monte Carlo simulation.
 */

#include <metal_stdlib>
#include "utils.metal"
using namespace metal;

/**
 * @brief Simulates the Brownian motion for a single particle
 *
 * @param t Pointer to output array for time points
 * @param x Pointer to output array for particle positions
 * @param start_position Initial position of the particle
 * @param diffusivity Diffusion coefficient (D) controlling the rate of diffusion
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param seed Random seed
 * @param idx Index of the particle
 */
inline void simulate(device float* t, device float* x, float start_position,
                        float diffusivity, float duration, float time_step,
                        ulong seed, uint idx) {
    float current_x = start_position;
    float current_t = 0.0f;
    
    t[0] = current_t;
    x[0] = current_x;
    
    float sigma = sqrt(2.0f * diffusivity * time_step);
    uint num_steps = uint(ceil(duration / time_step));
    
    PhiloxState state = philox_init(seed, idx);
    
    float xi;
    
    for (uint i = 0; i < num_steps - 1; ++i) {
        xi = philox_normal(state);
        current_x += xi * sigma;
        current_t += time_step;
        t[i + 1] = current_t;
        x[i + 1] = current_x;
    }
    
    float last_step = duration - current_t;
    xi = philox_normal(state);
    current_x += xi * sqrt(2.0f * diffusivity * last_step);
    
    t[num_steps] = duration;
    x[num_steps] = current_x;
}

/**
 * @brief Simulates the end position of Brownian motion for a single particle
 *
 * @param start_position Initial position of the particle
 * @param diffusivity Diffusion coefficient (D) controlling the rate of diffusion
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param seed Random seed
 * @param idx Index of the particle
 * @return End position of the particle
 */
inline float end(float start_position, float diffusivity, float duration, 
                    float time_step, ulong seed, uint idx) {
    float current_x = start_position;
    
    float sigma = sqrt(2.0f * diffusivity * time_step);
    uint num_steps = uint(ceil(duration / time_step));
    
    PhiloxState state = philox_init(seed, idx);
    
    float xi;
    
    for (uint i = 0; i < num_steps - 1; ++i) {
        xi = philox_normal(state);
        current_x += xi * sigma;
    }
    
    float last_step = duration - float(num_steps - 1) * time_step;
    xi = philox_normal(state);
    current_x += xi * sqrt(2.0f * diffusivity * last_step);
    
    return current_x;
}

/**
 * @brief Simulates Brownian motion and computes the mean position across all particles
 *
 * This kernel simulates multiple independent Brownian motion paths and computes
 * their mean position at the end of the simulation. Each thread handles one particle.
 *
 * @param out Pointer to output value (accumulated sum, should be divided by particles count in host)
 * @param start_position Initial position for all particles
 * @param diffusivity Diffusion coefficient (D) controlling the rate of diffusion
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param particles Number of particles to simulate
 * @param seed Random seed
 */
kernel void mean(device atomic_float* out [[buffer(0)]],
                    constant float& start_position [[buffer(1)]],
                    constant float& diffusivity [[buffer(2)]],
                    constant float& duration [[buffer(3)]],
                    constant float& time_step [[buffer(4)]],
                    constant uint& particles [[buffer(5)]],
                    constant ulong& seed [[buffer(6)]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint tg_size [[threads_per_threadgroup]],
                    uint idx [[thread_position_in_grid]],
                    threadgroup float* simd_sums [[threadgroup(0)]]) {
    
    float val = 0.0f;
    
    if (idx < particles) {
        val = end(start_position, diffusivity, duration, time_step, seed, idx);
    }
    
    float block_sum = threadgroup_reduce_sum(val, simd_sums, tid, tg_size);
    
    if (tid == 0) {
        atomic_fetch_add_explicit(out, block_sum, memory_order_relaxed);
    }
}

/**
 * @brief Computes the mean squared displacement (MSD) of Brownian motion
 *
 * This kernel simulates multiple independent Brownian motion paths and computes
 * their mean squared displacement at the end of the simulation.
 */
kernel void msd(device atomic_float* out [[buffer(0)]],
                   constant float& start_position [[buffer(1)]],
                   constant float& diffusivity [[buffer(2)]],
                   constant float& duration [[buffer(3)]],
                   constant float& time_step [[buffer(4)]],
                   constant uint& particles [[buffer(5)]],
                   constant ulong& seed [[buffer(6)]],
                   uint tid [[thread_position_in_threadgroup]],
                   uint tg_size [[threads_per_threadgroup]],
                   uint idx [[thread_position_in_grid]],
                   threadgroup float* simd_sums [[threadgroup(0)]]) {
    
    float val = 0.0f;
    
    if (idx < particles) {
        float end_position = end(0.0f, diffusivity, duration, time_step, seed, idx);
        val = (end_position - start_position) * (end_position - start_position);
    }
    
    float block_sum = threadgroup_reduce_sum(val, simd_sums, tid, tg_size);
    
    if (tid == 0) {
        atomic_fetch_add_explicit(out, block_sum, memory_order_relaxed);
    }
}

/**
 * @brief Computes the specified raw moment of Brownian motion
 *
 * This kernel simulates multiple independent Brownian motion paths and computes
 * their raw moment of specified integer order at the end of the simulation.
 * The raw moment is calculated as E[X^n].
 */
kernel void raw_moment(device atomic_float* out [[buffer(0)]],
                          constant float& start_position [[buffer(1)]],
                          constant float& diffusivity [[buffer(2)]],
                          constant int& order [[buffer(3)]],
                          constant float& duration [[buffer(4)]],
                          constant float& time_step [[buffer(5)]],
                          constant uint& particles [[buffer(6)]],
                          constant ulong& seed [[buffer(7)]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tg_size [[threads_per_threadgroup]],
                          uint idx [[thread_position_in_grid]],
                          threadgroup float* simd_sums [[threadgroup(0)]]) {
    
    float val = 0.0f;
    
    if (idx < particles) {
        float end_position = end(start_position, diffusivity, duration, time_step, seed, idx);
        val = powi(end_position, order);
    }
    
    float block_sum = threadgroup_reduce_sum(val, simd_sums, tid, tg_size);
    
    if (tid == 0) {
        atomic_fetch_add_explicit(out, block_sum, memory_order_relaxed);
    }
}

/**
 * @brief Computes the specified central moment of Brownian motion
 *
 * This kernel simulates multiple independent Brownian motion paths and computes
 * their central moment of specified integer order at the end of the simulation.
 * The central moment is calculated as E[(X-μ)^n] where μ is the mean.
 */
kernel void central_moment(device atomic_float* out [[buffer(0)]],
                              constant int& order [[buffer(1)]],
                              constant float& mean [[buffer(2)]],
                              constant float& start_position [[buffer(3)]],
                              constant float& diffusivity [[buffer(4)]],
                              constant float& duration [[buffer(5)]],
                              constant float& time_step [[buffer(6)]],
                              constant uint& particles [[buffer(7)]],
                              constant ulong& seed [[buffer(8)]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint tg_size [[threads_per_threadgroup]],
                              uint idx [[thread_position_in_grid]],
                              threadgroup float* simd_sums [[threadgroup(0)]]) {
    
    float val = 0.0f;
    
    if (idx < particles) {
        float end_position = end(start_position, diffusivity, duration, time_step, seed, idx);
        val = powi(end_position - mean, order);
    }
    
    float block_sum = threadgroup_reduce_sum(val, simd_sums, tid, tg_size);
    
    if (tid == 0) {
        atomic_fetch_add_explicit(out, block_sum, memory_order_relaxed);
    }
}

/**
 * @brief Computes the specified fractional raw moment of Brownian motion
 *
 * This kernel simulates multiple independent Brownian motion paths and computes
 * their raw moment of specified fractional order at the end of the simulation.
 * The fractional raw moment is calculated as E[|X|^r] where r is a real number.
 */
kernel void frac_raw_moment(device atomic_float* out [[buffer(0)]],
                               constant float& start_position [[buffer(1)]],
                               constant float& diffusivity [[buffer(2)]],
                               constant float& order [[buffer(3)]],
                               constant float& duration [[buffer(4)]],
                               constant float& time_step [[buffer(5)]],
                               constant uint& particles [[buffer(6)]],
                               constant ulong& seed [[buffer(7)]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tg_size [[threads_per_threadgroup]],
                               uint idx [[thread_position_in_grid]],
                               threadgroup float* simd_sums [[threadgroup(0)]]) {
    
    float val = 0.0f;
    
    if (idx < particles) {
        float end_position = end(start_position, diffusivity, duration, time_step, seed, idx);
        val = pow(abs(end_position), order);
    }
    
    float block_sum = threadgroup_reduce_sum(val, simd_sums, tid, tg_size);
    
    if (tid == 0) {
        atomic_fetch_add_explicit(out, block_sum, memory_order_relaxed);
    }
}

/**
 * @brief Computes the specified fractional central moment of Brownian motion
 *
 * This kernel simulates multiple independent Brownian motion paths and computes
 * their central moment of specified fractional order at the end of the simulation.
 * The fractional central moment is calculated as E[|X-μ|^r] where r is a real number.
 */
kernel void frac_central_moment(device atomic_float* out [[buffer(0)]],
                                   constant float& mean [[buffer(1)]],
                                   constant float& order [[buffer(2)]],
                                   constant float& start_position [[buffer(3)]],
                                   constant float& diffusivity [[buffer(4)]],
                                   constant float& duration [[buffer(5)]],
                                   constant float& time_step [[buffer(6)]],
                                   constant uint& particles [[buffer(7)]],
                                   constant ulong& seed [[buffer(8)]],
                                   uint tid [[thread_position_in_threadgroup]],
                                   uint tg_size [[threads_per_threadgroup]],
                                   uint idx [[thread_position_in_grid]],
                                   threadgroup float* simd_sums [[threadgroup(0)]]) {
    
    float val = 0.0f;
    
    if (idx < particles) {
        float end_position = end(start_position, diffusivity, duration, time_step, seed, idx);
        val = pow(abs(end_position - mean), order);
    }
    
    float block_sum = threadgroup_reduce_sum(val, simd_sums, tid, tg_size);
    
    if (tid == 0) {
        atomic_fetch_add_explicit(out, block_sum, memory_order_relaxed);
    }
}
