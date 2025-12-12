/**
 * Metal Kernel for Ornstein-Uhlenbeck Process Simulation
 *
 * The OU process is described by the SDE:
 * dX_t = -theta * X_t dt + sigma * dW_t
 *
 * where:
 * - theta: mean reversion speed
 * - sigma: volatility
 * - W_t: Wiener process
 */

#include <metal_stdlib>
#include "utils.metal"
using namespace metal;

/**
 * @brief Simulates Ornstein-Uhlenbeck process trajectories
 *
 * @param t Pointer to output array for time points
 * @param x Pointer to output array for particle positions
 * @param start_position Initial position for all particles
 * @param theta Mean reversion speed
 * @param sigma Volatility
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param seed Random seed
 * @param idx Index of the particle
 */
inline void simulate(device float* t, device float* x, float start_position,
                        float theta, float sigma, float duration, float time_step,
                        ulong seed, uint idx) {
    float current_x = start_position;
    float current_t = 0.0f;

    t[0] = current_t;
    x[0] = current_x;

    float scale = sqrt(sigma * time_step);
    uint num_steps = uint(ceil(duration / time_step));

    PhiloxState state = philox_init(seed, idx);

    float mu;
    float xi;

    for (uint i = 0; i < num_steps - 1; ++i) {
        mu = -theta * current_x;
        xi = philox_normal(state);
        current_x += mu * time_step + scale * xi;
        current_t += time_step;

        t[i + 1] = current_t;
        x[i + 1] = current_x;
    }

    float last_step = duration - current_t;
    xi = philox_normal(state);
    mu = -theta * current_x;
    current_x += mu * last_step + sqrt(sigma * last_step) * xi;

    t[num_steps] = duration;
    x[num_steps] = current_x;
}

/**
 * @brief Simulates the end position of Ornstein-Uhlenbeck process
 */
inline float end(float start_position, float theta, float sigma,
                    float duration, float time_step, ulong seed, uint idx) {
    float current_x = start_position;

    float scale = sqrt(sigma * time_step);
    uint num_steps = uint(ceil(duration / time_step));

    PhiloxState state = philox_init(seed, idx);

    float mu;
    float xi;

    for (uint i = 0; i < num_steps - 1; ++i) {
        mu = -theta * current_x;
        xi = philox_normal(state);
        current_x += mu * time_step + scale * xi;
    }

    float last_step = duration - float(num_steps - 1) * time_step;
    xi = philox_normal(state);
    mu = -theta * current_x;
    current_x += mu * last_step + sqrt(sigma * last_step) * xi;

    return current_x - start_position;
}

/**
 * @brief Computes the mean position of Ornstein-Uhlenbeck process
 */
kernel void mean(device atomic_float* out [[buffer(0)]],
                    constant float& start_position [[buffer(1)]],
                    constant float& theta [[buffer(2)]],
                    constant float& sigma [[buffer(3)]],
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
        val = end(start_position, theta, sigma, duration, time_step, seed, idx);
    }

    float block_sum = threadgroup_reduce_sum(val, simd_sums, tid, tg_size);

    if (tid == 0) {
        atomic_fetch_add_explicit(out, block_sum, memory_order_relaxed);
    }
}

/**
 * @brief Computes the mean squared displacement (MSD) of Ornstein-Uhlenbeck process
 */
kernel void msd(device atomic_float* out [[buffer(0)]],
                   constant float& start_position [[buffer(1)]],
                   constant float& theta [[buffer(2)]],
                   constant float& sigma [[buffer(3)]],
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
        float end_position = end(start_position, theta, sigma, duration, time_step, seed, idx);
        val = (end_position - start_position) * (end_position - start_position);
    }

    float block_sum = threadgroup_reduce_sum(val, simd_sums, tid, tg_size);

    if (tid == 0) {
        atomic_fetch_add_explicit(out, block_sum, memory_order_relaxed);
    }
}

/**
 * @brief Computes the raw moment of Ornstein-Uhlenbeck process
 */
kernel void raw_moment(device atomic_float* out [[buffer(0)]],
                          constant float& start_position [[buffer(1)]],
                          constant float& theta [[buffer(2)]],
                          constant float& sigma [[buffer(3)]],
                          constant int& order [[buffer(4)]],
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
        float end_position = end(start_position, theta, sigma, duration, time_step, seed, idx);
        val = powi(end_position, order);
    }

    float block_sum = threadgroup_reduce_sum(val, simd_sums, tid, tg_size);

    if (tid == 0) {
        atomic_fetch_add_explicit(out, block_sum, memory_order_relaxed);
    }
}

/**
 * @brief Computes the central moment of Ornstein-Uhlenbeck process
 */
kernel void central_moment(device atomic_float* out [[buffer(0)]],
                              constant int& order [[buffer(1)]],
                              constant float& mean [[buffer(2)]],
                              constant float& start_position [[buffer(3)]],
                              constant float& theta [[buffer(4)]],
                              constant float& sigma [[buffer(5)]],
                              constant float& duration [[buffer(6)]],
                              constant float& time_step [[buffer(7)]],
                              constant uint& particles [[buffer(8)]],
                              constant ulong& seed [[buffer(9)]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint tg_size [[threads_per_threadgroup]],
                              uint idx [[thread_position_in_grid]],
                              threadgroup float* simd_sums [[threadgroup(0)]]) {

    float val = 0.0f;

    if (idx < particles) {
        float end_position = end(start_position, theta, sigma, duration, time_step, seed, idx);
        val = powi(end_position - mean, order);
    }

    float block_sum = threadgroup_reduce_sum(val, simd_sums, tid, tg_size);

    if (tid == 0) {
        atomic_fetch_add_explicit(out, block_sum, memory_order_relaxed);
    }
}

/**
 * @brief Computes the fractional raw moment of Ornstein-Uhlenbeck process
 */
kernel void frac_raw_moment(device atomic_float* out [[buffer(0)]],
                               constant float& start_position [[buffer(1)]],
                               constant float& theta [[buffer(2)]],
                               constant float& sigma [[buffer(3)]],
                               constant float& order [[buffer(4)]],
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
        float end_position = end(start_position, theta, sigma, duration, time_step, seed, idx);
        val = pow(abs(end_position), order);
    }

    float block_sum = threadgroup_reduce_sum(val, simd_sums, tid, tg_size);

    if (tid == 0) {
        atomic_fetch_add_explicit(out, block_sum, memory_order_relaxed);
    }
}

/**
 * @brief Computes the fractional central moment of Ornstein-Uhlenbeck process
 */
kernel void frac_central_moment(device atomic_float* out [[buffer(0)]],
                                   constant float& mean [[buffer(1)]],
                                   constant float& order [[buffer(2)]],
                                   constant float& start_position [[buffer(3)]],
                                   constant float& theta [[buffer(4)]],
                                   constant float& sigma [[buffer(5)]],
                                   constant float& duration [[buffer(6)]],
                                   constant float& time_step [[buffer(7)]],
                                   constant uint& particles [[buffer(8)]],
                                   constant ulong& seed [[buffer(9)]],
                                   uint tid [[thread_position_in_threadgroup]],
                                   uint tg_size [[threads_per_threadgroup]],
                                   uint idx [[thread_position_in_grid]],
                                   threadgroup float* simd_sums [[threadgroup(0)]]) {

    float val = 0.0f;

    if (idx < particles) {
        float end_position = end(start_position, theta, sigma, duration, time_step, seed, idx);
        val = pow(abs(end_position - mean), order);
    }

    float block_sum = threadgroup_reduce_sum(val, simd_sums, tid, tg_size);

    if (tid == 0) {
        atomic_fetch_add_explicit(out, block_sum, memory_order_relaxed);
    }
}
