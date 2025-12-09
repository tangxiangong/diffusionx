/**
 * Metal Kernel for Random Number Generation
 *
 * This file implements various Metal compute kernels for generating
 * random numbers from different distributions.
 */

#include <metal_stdlib>
#include "utils.metal"
using namespace metal;

/**
 * @brief Generate exponential random numbers
 */
kernel void randexp(device float* out [[buffer(0)]],
                    constant uint& len [[buffer(1)]],
                    constant ulong& seed [[buffer(2)]],
                    uint idx [[thread_position_in_grid]]) {
    if (idx < len) {
        PhiloxState state = philox_init(seed, idx);
        float u = philox_uniform(state);
        out[idx] = -log(u);
    }
}

/**
 * @brief Sample from standard stable distribution when α = 1
 */
inline float sample_standard_alpha_one(float alpha, float beta, thread PhiloxState& state) {
    float v = philox_uniform(state) * M_PI_F - M_PI_2_F;
    float w = -log(philox_uniform(state));
    float half_pi_plus_beta_v = M_PI_2_F + beta * v;
    float c1 = half_pi_plus_beta_v * tan(v);
    float c2 = (M_PI_2_F * w * cos(v)) / log(half_pi_plus_beta_v) * beta;
    return (c1 - c2) * M_2_PI_F;
}

/**
 * @brief Sample from standard stable distribution when α ≠ 1
 */
inline float sample_standard_alpha_with_constants(
    float alpha, float inv_alpha, float one_minus_alpha_div_alpha,
    float b, float s, thread PhiloxState& state) {
    float v = philox_uniform(state) * M_PI_F - M_PI_2_F;
    float w = -log(philox_uniform(state));
    float v_plus_b = v + b;
    float cos_v = cos(v);
    float c1 = alpha * sin(v_plus_b) / pow(cos_v, inv_alpha);
    float c2 = pow(cos(v - alpha * v_plus_b) / w, one_minus_alpha_div_alpha);
    return s * c1 * c2;
}

/**
 * @brief Generate standard stable random numbers
 */
kernel void standard_stable_rand(device float* out [[buffer(0)]],
                                 constant float& alpha [[buffer(1)]],
                                 constant float& beta [[buffer(2)]],
                                 constant float& inv_alpha [[buffer(3)]],
                                 constant float& one_minus_alpha_div_alpha [[buffer(4)]],
                                 constant float& b [[buffer(5)]],
                                 constant float& s [[buffer(6)]],
                                 constant uint& len [[buffer(7)]],
                                 constant ulong& seed [[buffer(8)]],
                                 uint idx [[thread_position_in_grid]]) {
    if (idx < len) {
        PhiloxState state = philox_init(seed, idx);
        float r;
        if (abs(alpha - 1.0f) < 1e-3f) {
            r = sample_standard_alpha_one(alpha, beta, state);
        } else {
            r = sample_standard_alpha_with_constants(
                alpha, inv_alpha, one_minus_alpha_div_alpha, b, s, state);
        }
        out[idx] = r;
    }
}

/**
 * @brief Generate uniform random numbers in (0, 1]
 */
kernel void randuniform(device float* out [[buffer(0)]],
                        constant uint& len [[buffer(1)]],
                        constant ulong& seed [[buffer(2)]],
                        uint idx [[thread_position_in_grid]]) {
    if (idx < len) {
        PhiloxState state = philox_init(seed, idx);
        out[idx] = philox_uniform(state);
    }
}

/**
 * @brief Generate normal random numbers
 */
kernel void randnormal(device float* out [[buffer(0)]],
                       constant uint& len [[buffer(1)]],
                       constant float& mu [[buffer(2)]],
                       constant float& sigma [[buffer(3)]],
                       constant ulong& seed [[buffer(4)]],
                       uint idx [[thread_position_in_grid]]) {
    if (idx < len) {
        PhiloxState state = philox_init(seed, idx);
        out[idx] = mu + sigma * philox_normal(state);
    }
}
