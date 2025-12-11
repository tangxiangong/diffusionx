/**
 * Metal Kernel for alpha stable Levy process (levy.rs)
 *
 * Corresponds to: src/simulation/continuous/levy.rs and src/gpu/metal/levy.rs
 *
 * This file implements various Metal compute kernels for computing statistical
 * properties of Levy processes through Monte Carlo simulation.
 */

#include <metal_stdlib>
#include "utils.metal"
using namespace metal;

/**
 * @brief Sample from symmetric standard stable distribution when α = 1 (Cauchy)
 */
inline float sample_symmetric_standard_alpha_one(thread PhiloxState& state) {
    float v = philox_uniform(state) * M_PI_F - M_PI_2_F;
    float w = -log(philox_uniform(state));
    float c = M_PI_2_F * tan(v);
    return c * M_2_PI_F;
}

/**
 * @brief Sample from symmetric standard stable distribution when α ≠ 1
 */
inline float sample_symmetric_standard_alpha_with_constants(
    float alpha, float inv_alpha, float one_minus_alpha_div_alpha,
    thread PhiloxState& state) {
    float v = philox_uniform(state) * M_PI_F - M_PI_2_F;
    float w = -log(philox_uniform(state));
    float cos_v = cos(v);
    float c1 = alpha * sin(v) / pow(cos_v, inv_alpha);
    float c2 = pow(cos(v - alpha * v) / w, one_minus_alpha_div_alpha);
    return c1 * c2;
}

/**
 * @brief Simulates the α stable Levy process for a single particle when α = 1
 */
inline void simulate_alpha_one(device float* t, device float* x, float start_position,
                                    float duration, float time_step, ulong seed, uint idx) {
    float current_x = start_position;
    float current_t = 0.0f;
    
    t[0] = current_t;
    x[0] = current_x;
    
    float sigma = time_step;
    uint num_steps = uint(ceil(duration / time_step));
    
    PhiloxState state = philox_init(seed, idx);
    
    float xi;
    
    for (uint i = 0; i < num_steps - 1; ++i) {
        xi = sample_symmetric_standard_alpha_one(state);
        current_x += xi * sigma;
        current_t += time_step;
        t[i + 1] = current_t;
        x[i + 1] = current_x;
    }
    
    float last_step = duration - current_t;
    xi = sample_symmetric_standard_alpha_one(state);
    current_x += xi * last_step;
    
    t[num_steps] = duration;
    x[num_steps] = current_x;
}

/**
 * @brief Simulates the α stable Levy process for a single particle when α ≠ 1
 */
inline void simulate_alpha(device float* t, device float* x, float alpha,
                                float start_position, float duration, float time_step,
                                ulong seed, uint idx, float inv_alpha,
                                float one_minus_alpha_div_alpha) {
    float current_x = start_position;
    float current_t = 0.0f;
    
    t[0] = current_t;
    x[0] = current_x;
    
    float sigma = pow(time_step, inv_alpha);
    uint num_steps = uint(ceil(duration / time_step));
    
    PhiloxState state = philox_init(seed, idx);
    
    float xi;
    
    for (uint i = 0; i < num_steps - 1; ++i) {
        xi = sample_symmetric_standard_alpha_with_constants(
            alpha, inv_alpha, one_minus_alpha_div_alpha, state);
        current_x += xi * sigma;
        current_t += time_step;
        t[i + 1] = current_t;
        x[i + 1] = current_x;
    }
    
    float last_step = duration - current_t;
    xi = sample_symmetric_standard_alpha_with_constants(
        alpha, inv_alpha, one_minus_alpha_div_alpha, state);
    sigma = pow(last_step, inv_alpha);
    current_x += xi * sigma;
    
    t[num_steps] = duration;
    x[num_steps] = current_x;
}

/**
 * @brief Simulates the α stable Levy process for a single particle
 */
inline void simulate(device float* t, device float* x, float alpha,
                          float start_position, float duration, float time_step,
                          ulong seed, uint idx, float inv_alpha,
                          float one_minus_alpha_div_alpha) {
    if (abs(alpha - 1.0f) < 1e-6f) {
        simulate_alpha_one(t, x, start_position, duration, time_step, seed, idx);
    } else {
        simulate_alpha(t, x, alpha, start_position, duration, time_step, seed, idx,
                            inv_alpha, one_minus_alpha_div_alpha);
    }
}

/**
 * @brief Simulates the end position of α stable Levy process when α = 1
 */
inline float end_alpha_one(float start_position, float duration,
                                float time_step, ulong seed, uint idx) {
    float current_x = start_position;
    float xi;
    float sigma = time_step;
    uint num_steps = uint(ceil(duration / time_step));
    
    PhiloxState state = philox_init(seed, idx);
    
    for (uint i = 0; i < num_steps - 1; ++i) {
        xi = sample_symmetric_standard_alpha_one(state);
        current_x += xi * sigma;
    }
    
    float last_step = duration - float(num_steps - 1) * time_step;
    xi = sample_symmetric_standard_alpha_one(state);
    current_x += xi * last_step;
    
    return current_x;
}

/**
 * @brief Simulates the end position of α stable Levy process when α ≠ 1
 */
inline float end_alpha(float alpha, float start_position,
                            float duration, float time_step,
                            ulong seed, uint idx,
                            float inv_alpha, float one_minus_alpha_div_alpha) {
    float current_x = start_position;
    float xi;
    float sigma = pow(time_step, inv_alpha);
    uint num_steps = uint(ceil(duration / time_step));
    
    PhiloxState state = philox_init(seed, idx);
    
    for (uint i = 0; i < num_steps - 1; ++i) {
        xi = sample_symmetric_standard_alpha_with_constants(
            alpha, inv_alpha, one_minus_alpha_div_alpha, state);
        current_x += xi * sigma;
    }
    
    float last_step = duration - float(num_steps - 1) * time_step;
    xi = sample_symmetric_standard_alpha_with_constants(
        alpha, inv_alpha, one_minus_alpha_div_alpha, state);
    sigma = pow(last_step, inv_alpha);
    current_x += xi * sigma;
    
    return current_x;
}

/**
 * @brief Simulates the end position of α stable Levy process
 */
inline float end(float alpha, float start_position, float duration,
                      float time_step, ulong seed, uint idx,
                      float inv_alpha, float one_minus_alpha_div_alpha) {
    if (abs(alpha - 1.0f) < 1e-6f) {
        return end_alpha_one(start_position, duration, time_step, seed, idx);
    } else {
        return end_alpha(alpha, start_position, duration, time_step, seed, idx,
                              inv_alpha, one_minus_alpha_div_alpha);
    }
}

/**
 * @brief Computes the mean position of Levy process
 */
kernel void mean(device atomic_float* out [[buffer(0)]],
                      constant float& alpha [[buffer(1)]],
                      constant float& start_position [[buffer(2)]],
                      constant float& duration [[buffer(3)]],
                      constant float& time_step [[buffer(4)]],
                      constant float& inv_alpha [[buffer(5)]],
                      constant float& one_minus_alpha_div_alpha [[buffer(6)]],
                      constant uint& particles [[buffer(7)]],
                      constant ulong& seed [[buffer(8)]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint tg_size [[threads_per_threadgroup]],
                      uint idx [[thread_position_in_grid]],
                      threadgroup float* simd_sums [[threadgroup(0)]]) {
    
    float val = 0.0f;
    
    if (idx < particles) {
        val = end(alpha, start_position, duration, time_step, seed, idx,
                       inv_alpha, one_minus_alpha_div_alpha);
    }
    
    float block_sum = threadgroup_reduce_sum(val, simd_sums, tid, tg_size);
    
    if (tid == 0) {
        atomic_fetch_add_explicit(out, block_sum, memory_order_relaxed);
    }
}

/**
 * @brief Computes the raw moment of Levy process
 */
kernel void raw_moment(device atomic_float* out [[buffer(0)]],
                            constant float& alpha [[buffer(1)]],
                            constant float& start_position [[buffer(2)]],
                            constant int& order [[buffer(3)]],
                            constant float& duration [[buffer(4)]],
                            constant float& time_step [[buffer(5)]],
                            constant float& inv_alpha [[buffer(6)]],
                            constant float& one_minus_alpha_div_alpha [[buffer(7)]],
                            constant uint& particles [[buffer(8)]],
                            constant ulong& seed [[buffer(9)]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint tg_size [[threads_per_threadgroup]],
                            uint idx [[thread_position_in_grid]],
                            threadgroup float* simd_sums [[threadgroup(0)]]) {
    
    float val = 0.0f;
    
    if (idx < particles) {
        float end_position = end(alpha, start_position, duration, time_step, seed, idx,
                                      inv_alpha, one_minus_alpha_div_alpha);
        val = powi(end_position, order);
    }
    
    float block_sum = threadgroup_reduce_sum(val, simd_sums, tid, tg_size);
    
    if (tid == 0) {
        atomic_fetch_add_explicit(out, block_sum, memory_order_relaxed);
    }
}

/**
 * @brief Computes the fractional raw moment of Levy process
 */
kernel void frac_raw_moment(device atomic_float* out [[buffer(0)]],
                                 constant float& alpha [[buffer(1)]],
                                 constant float& start_position [[buffer(2)]],
                                 constant float& order [[buffer(3)]],
                                 constant float& duration [[buffer(4)]],
                                 constant float& time_step [[buffer(5)]],
                                 constant float& inv_alpha [[buffer(6)]],
                                 constant float& one_minus_alpha_div_alpha [[buffer(7)]],
                                 constant uint& particles [[buffer(8)]],
                                 constant ulong& seed [[buffer(9)]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint tg_size [[threads_per_threadgroup]],
                                 uint idx [[thread_position_in_grid]],
                                 threadgroup float* simd_sums [[threadgroup(0)]]) {
    
    float val = 0.0f;
    
    if (idx < particles) {
        float end_position = end(alpha, start_position, duration, time_step, seed, idx,
                                      inv_alpha, one_minus_alpha_div_alpha);
        val = pow(abs(end_position), order);
    }
    
    float block_sum = threadgroup_reduce_sum(val, simd_sums, tid, tg_size);
    
    if (tid == 0) {
        atomic_fetch_add_explicit(out, block_sum, memory_order_relaxed);
    }
}

/**
 * @brief Computes the fractional central moment of Levy process
 */
kernel void frac_central_moment(device atomic_float* out [[buffer(0)]],
                                     constant float& mean [[buffer(1)]],
                                     constant float& alpha [[buffer(2)]],
                                     constant float& start_position [[buffer(3)]],
                                     constant float& order [[buffer(4)]],
                                     constant float& duration [[buffer(5)]],
                                     constant float& time_step [[buffer(6)]],
                                     constant float& inv_alpha [[buffer(7)]],
                                     constant float& one_minus_alpha_div_alpha [[buffer(8)]],
                                     constant uint& particles [[buffer(9)]],
                                     constant ulong& seed [[buffer(10)]],
                                     uint tid [[thread_position_in_threadgroup]],
                                     uint tg_size [[threads_per_threadgroup]],
                                     uint idx [[thread_position_in_grid]],
                                     threadgroup float* simd_sums [[threadgroup(0)]]) {
    
    float val = 0.0f;
    
    if (idx < particles) {
        float end_position = end(alpha, start_position, duration, time_step, seed, idx,
                                      inv_alpha, one_minus_alpha_div_alpha);
        val = pow(abs(end_position - mean), order);
    }
    
    float block_sum = threadgroup_reduce_sum(val, simd_sums, tid, tg_size);
    
    if (tid == 0) {
        atomic_fetch_add_explicit(out, block_sum, memory_order_relaxed);
    }
}
