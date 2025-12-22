/**
 * Metal Utility Functions for GPU Kernels
 *
 * This file provides common utility functions for parallel reduction
 * operations in Metal compute shaders.
 */

#include <metal_stdlib>
using namespace metal;

// Constants
constant uint SIMD_SIZE = 32;

/**
 * @brief Performs a warp-level (SIMD group) sum reduction
 *
 * Uses Metal's simd_shuffle_down to efficiently reduce values within a SIMD group.
 *
 * @param val The value to reduce
 * @return The sum of all values in the SIMD group (valid only in lane 0)
 */
inline float simd_reduce_sum(float val) {
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset >>= 1) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}

/**
 * @brief Performs a threadgroup-level (block) sum reduction
 *
 * First reduces within each SIMD group, then combines the results.
 *
 * @param val The value to reduce
 * @param simd_sums Shared memory array for storing SIMD group sums
 * @param tid Thread index within the threadgroup
 * @param tg_size Size of the threadgroup
 * @return The sum of all values in the threadgroup (valid only in thread 0)
 */
inline float threadgroup_reduce_sum(float val, threadgroup float* simd_sums, uint tid, uint tg_size) {
    // Reduce within SIMD group
    val = simd_reduce_sum(val);

    uint lane = tid % SIMD_SIZE;
    uint simd_id = tid / SIMD_SIZE;

    // First lane of each SIMD group stores its sum
    if (lane == 0) {
        simd_sums[simd_id] = val;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First SIMD group reduces all SIMD sums
    float block_sum = 0.0f;
    if (simd_id == 0) {
        uint num_simds = (tg_size + SIMD_SIZE - 1) / SIMD_SIZE;
        if (tid < num_simds) {
            block_sum = simd_sums[lane];
        }
        block_sum = simd_reduce_sum(block_sum);
    }

    return block_sum;
}

/**
 * @brief Philox 4x32 random number generator state
 *
 * A counter-based RNG that provides high-quality random numbers
 * suitable for Monte Carlo simulations.
 */
struct PhiloxState {
    uint4 counter;
    uint2 key;
};

/**
 * @brief Single round of Philox 4x32
 */
inline uint4 philox_round(uint4 ctr, uint2 key) {
    constexpr uint PHILOX_M4x32_0 = 0xD2511F53;
    constexpr uint PHILOX_M4x32_1 = 0xCD9E8D57;

    uint hi0 = mulhi(PHILOX_M4x32_0, ctr.x);
    uint lo0 = PHILOX_M4x32_0 * ctr.x;
    uint hi1 = mulhi(PHILOX_M4x32_1, ctr.z);
    uint lo1 = PHILOX_M4x32_1 * ctr.z;

    return uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
}

/**
 * @brief Bump the Philox key
 */
inline uint2 philox_bump_key(uint2 key) {
    constexpr uint PHILOX_W32_0 = 0x9E3779B9;
    constexpr uint PHILOX_W32_1 = 0xBB67AE85;
    return uint2(key.x + PHILOX_W32_0, key.y + PHILOX_W32_1);
}

/**
 * @brief Initialize Philox state
 */
inline PhiloxState philox_init(ulong seed, uint idx) {
    PhiloxState state;
    state.counter = uint4(idx, 0, 0, 0);
    state.key = uint2(uint(seed), uint(seed >> 32));
    return state;
}

/**
 * @brief Generate 4 random uint32 values using Philox 4x32-10
 */
inline uint4 philox_generate(thread PhiloxState& state) {
    uint4 ctr = state.counter;
    uint2 key = state.key;

    // 10 rounds
    for (int i = 0; i < 10; i++) {
        ctr = philox_round(ctr, key);
        key = philox_bump_key(key);
    }

    // Increment counter
    state.counter.x += 1;
    if (state.counter.x == 0) {
        state.counter.y += 1;
        if (state.counter.y == 0) {
            state.counter.z += 1;
            if (state.counter.z == 0) {
                state.counter.w += 1;
            }
        }
    }

    return ctr;
}

/**
 * @brief Generate a uniform random float in (0, 1]
 */
inline float philox_uniform(thread PhiloxState& state) {
    uint4 r = philox_generate(state);
    // Convert to float in (0, 1]
    return (float(r.x) + 1.0f) * (1.0f / 4294967296.0f);
}

/**
 * @brief Generate a standard normal random float using Box-Muller transform
 */
inline float philox_normal(thread PhiloxState& state) {
    float u1 = philox_uniform(state);
    float u2 = philox_uniform(state);

    float r = sqrt(-2.0f * log(u1));
    float theta = 2.0f * M_PI_F * u2;

    return r * cos(theta);
}

/**
 * @brief Simple Xorshift RNG state for basic random number generation
 */
struct XorshiftState {
    uint state;
};

/**
 * @brief Initialize Xorshift state
 */
inline XorshiftState xorshift_init(ulong seed, uint idx) {
    XorshiftState state;
    state.state = uint(seed) ^ (idx * 1664525u + 1013904223u);
    if (state.state == 0) state.state = 1;
    return state;
}

/**
 * @brief Generate a random uint32 using Xorshift
 */
inline uint xorshift_generate(thread XorshiftState& state) {
    uint x = state.state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state.state = x;
    return x;
}

/**
 * @brief Generate a uniform random float in (0, 1] using Xorshift
 */
inline float xorshift_uniform(thread XorshiftState& state) {
    return (float(xorshift_generate(state)) + 1.0f) * (1.0f / 4294967296.0f);
}

/**
 * @brief Generate a standard normal random float using Box-Muller transform with Xorshift
 */
inline float xorshift_normal(thread XorshiftState& state) {
    float u1 = xorshift_uniform(state);
    float u2 = xorshift_uniform(state);

    float r = sqrt(-2.0f * log(u1));
    float theta = 2.0f * M_PI_F * u2;

    return r * cos(theta);
}

/**
 * @brief Compute integer power of a float
 *
 * This function correctly handles negative bases with integer exponents,
 * unlike pow(float, float) which returns NaN for negative bases.
 *
 * @param base The base value (can be negative)
 * @param exp The integer exponent
 * @return base^exp
 */
inline float powi(float base, int exp) {
    if (exp == 0) return 1.0f;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    bool negative_exp = exp < 0;
    if (negative_exp) exp = -exp;

    float result = 1.0f;
    float current = base;

    while (exp > 0) {
        if (exp & 1) {
            result *= current;
        }
        current *= current;
        exp >>= 1;
    }

    return negative_exp ? (1.0f / result) : result;
}
