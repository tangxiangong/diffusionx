/**
 * CUDA Kernel for alpha stable Levy process (levy.rs)
 *
 * Corresponds to: src/simulation/continuous/levy.rs and src/gpu/levy.rs
 *
 * This file implements various CUDA kernels for computing statistical
 * properties of Levy processes through Monte Carlo simulation.
 */

#include <curand_kernel.h>
#include <math_constants.h>
#include "utils.cu"

QUALIFIERS float
sample_symmetric_standard_alpha_one(curandStatePhilox4_32_10_t *state)
{
    float v = curand_uniform(state) * CUDART_PI_F - CUDART_PIO2_F;
    float w = -logf(curand_uniform(state));
    float c = CUDART_PIO2_F * tanf(v);
    return c * CUDART_2_OVER_PI_F;
}

QUALIFIERS float sample_symmetric_standard_alpha_with_constants(
    float alpha, float inv_alpha, float one_minus_alpha_div_alpha,
    curandStatePhilox4_32_10_t *state)
{
    float v = curand_uniform(state) * CUDART_PI_F - CUDART_PIO2_F;
    float w = -logf(curand_uniform(state));
    float cos_v = cosf(v);
    float c1 = alpha * sinf(v) / powf(cos_v, inv_alpha);
    float c2 = powf(cosf(v - alpha * v) / w, one_minus_alpha_div_alpha);
    return c1 * c2;
}

/**
 * @brief Simulates the α stable Levy process for a single particle when α = 1
 *
 * @param t Pointer to array of time points
 * @param x Pointer to array of positions
 * @param start_position Initial position of the particle
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param seed Random seed for CURAND
 * @param idx Index of the particle
 */
QUALIFIERS void simulate_alpha_one(float *t, float *x, float start_position, float duration,
                                   float time_step, unsigned long long seed,
                                   size_t idx)
{
    float current_x = start_position;
    float current_t = 0.0f;

    t[0] = current_t;
    x[0] = current_x;

    float sigma = time_step;
    size_t num_steps = static_cast<size_t>(ceil(duration / time_step));

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    float xi;

    for (size_t i = 0; i < num_steps - 1; ++i)
    {
        xi = sample_symmetric_standard_alpha_one(&state);
        current_x += xi * sigma;
        current_t += time_step;
        t[i + 1] = current_t;
        x[i + 1] = current_x;
    }

    float last_step = duration - current_t;
    xi = sample_symmetric_standard_alpha_one(&state);
    sigma = last_step;
    current_x += xi * last_step;

    t[num_steps] = duration;
    x[num_steps] = current_x;
}

/**
 * @brief Simulates the α stable Levy process for a single particle when α != 1
 *
 * @param t Pointer to array of time points
 * @param x Pointer to array of positions
 * @param start_position Initial position of the particle
 * @param alpha Stability parameter (0 < α ≤ 2, α ≠ 1)
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param seed Random seed for CURAND
 * @param idx Index of the particle
 * @param inv_alpha Precomputed value of 1/α
 * @param one_minus_alpha_div_alpha Precomputed value of (1-α)/α
 */
QUALIFIERS void simulate_alpha(float *t, float *x, float alpha, float start_position,
                               float duration, float time_step,
                               unsigned long long seed, size_t idx,
                               float inv_alpha,
                               float one_minus_alpha_div_alpha)
{
    float current_x = start_position;
    float current_t = 0.0f;

    t[0] = current_t;
    x[0] = current_x;

    float sigma = powf(time_step, inv_alpha);
    size_t num_steps = static_cast<size_t>(ceil(duration / time_step));

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    float xi;

    for (size_t i = 0; i < num_steps - 1; ++i)
    {
        xi = sample_symmetric_standard_alpha_with_constants(
            alpha, inv_alpha, one_minus_alpha_div_alpha, &state);
        current_x += xi * sigma;
        current_t += time_step;
        t[i + 1] = current_t;
        x[i + 1] = current_x;
    }

    float last_step = duration - current_t;
    xi = sample_symmetric_standard_alpha_with_constants(
        alpha, inv_alpha, one_minus_alpha_div_alpha, &state);
    sigma = powf(last_step, inv_alpha);
    current_x += xi * sigma;

    t[num_steps] = duration;
    x[num_steps] = current_x;
}

/**
 * @brief Simulates the α stable Levy process for a single particle
 *
 * @param t Pointer to array of time points
 * @param x Pointer to array of positions
 * @param alpha Stability parameter (0 < α ≤ 2)
 * @param start_position Initial position of the particle
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param seed Random seed for CURAND
 * @param idx Index of the particle
 * @param inv_alpha Precomputed value of 1/α
 * @param one_minus_alpha_div_alpha Precomputed value of (1-α)/α
 */
QUALIFIERS void simulate(float *t, float *x, float alpha, float start_position, float duration,
                         float time_step, unsigned long long seed, size_t idx,
                         float inv_alpha, float one_minus_alpha_div_alpha)
{
    if (alpha == 1.0f)
    {
        return simulate_alpha_one(t, x, start_position, duration, time_step, seed, idx);
    }
    else
    {
        return simulate_alpha(t, x, alpha, start_position, duration, time_step, seed, idx,
                              inv_alpha, one_minus_alpha_div_alpha);
    }
}

/**
 * @brief Simulates the end position of α stable Levy process for a single particle when α = 1
 *
 * @param start_position Initial position of the particle
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param seed Random seed for CURAND
 * @param idx Index of the particle
 */
QUALIFIERS float end_alpha_one(float start_position, float duration,
                               float time_step, unsigned long long seed,
                               size_t idx)
{
    float current_x = start_position;
    float xi;
    float sigma = time_step;
    size_t num_steps = static_cast<size_t>(ceil(duration / time_step));

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    for (size_t i = 0; i < num_steps - 1; ++i)
    {
        xi = sample_symmetric_standard_alpha_one(&state);
        current_x += xi * sigma;
    }

    float last_step = duration - (num_steps - 1) * time_step;
    xi = sample_symmetric_standard_alpha_one(&state);
    sigma = last_step;
    current_x += xi * last_step;
    return current_x;
}

/**
 * @brief Simulates the end position of α stable Levy process for a single particle when α != 1
 *
 * @param start_position Initial position of the particle
 * @param alpha Stability parameter (0 < α ≤ 2, α ≠ 1)
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param seed Random seed for CURAND
 * @param idx Index of the particle
 * @param inv_alpha Precomputed value of 1/α
 * @param one_minus_alpha_div_alpha Precomputed value of (1-α)/α
 */
QUALIFIERS float end_alpha(float alpha, float start_position,
                           float duration, float time_step,
                           unsigned long long seed, size_t idx,
                           float inv_alpha,
                           float one_minus_alpha_div_alpha)
{
    float current_x = start_position;
    float xi;
    float sigma = powf(time_step, inv_alpha);
    size_t num_steps = static_cast<size_t>(ceil(duration / time_step));

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    for (size_t i = 0; i < num_steps - 1; ++i)
    {
        xi = sample_symmetric_standard_alpha_with_constants(
            alpha, inv_alpha, one_minus_alpha_div_alpha, &state);
        current_x += xi * sigma;
    }

    float last_step = duration - (num_steps - 1) * time_step;
    xi = sample_symmetric_standard_alpha_with_constants(
        alpha, inv_alpha, one_minus_alpha_div_alpha, &state);
    sigma = powf(last_step, inv_alpha);
    current_x += xi * sigma;
    return current_x;
}

/**
 * @brief Simulates the end position of α stable Levy process for a single particle
 *
 * @param alpha Stability parameter (0 < α ≤ 2)
 * @param start_position Initial position of the particle
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param seed Random seed for CURAND
 * @param idx Index of the particle
 * @param inv_alpha Precomputed value of 1/α
 * @param one_minus_alpha_div_alpha Precomputed value of (1-α)/α
 */
QUALIFIERS float end(float alpha, float start_position, float duration,
                     float time_step, unsigned long long seed, size_t idx,
                     float inv_alpha, float one_minus_alpha_div_alpha)
{
    if (alpha == 1.0f)
    {
        return end_alpha_one(start_position, duration, time_step, seed, idx);
    }
    else
    {
        return end_alpha(alpha, start_position, duration, time_step, seed, idx,
                         inv_alpha, one_minus_alpha_div_alpha);
    }
}

/**
 * @brief Simulates Levy process and computes the mean position across all
 * particles
 *
 * This kernel simulates multiple independent Levy process paths and computes
 * their mean position at the end of the simulation. Each thread handles one
 * particle.
 *
 * @param out Pointer to output value (accumulated sum, should be divided by
 * particles count in host)
 * @param alpha Stability parameter (0 < α ≤ 2)
 * @param start_position Initial position for all particles
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param particles Number of particles to simulate
 * @param seed Random seed for CURAND
 */
extern "C" __global__ void mean(float *out, float alpha, float start_position,
                                float duration, float time_step,
                                float inv_alpha,
                                float one_minus_alpha_div_alpha,
                                size_t particles, unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 0.0f;

    if (idx < particles)
    {
        val = end(alpha, start_position, duration, time_step, seed, idx,
                  inv_alpha, one_minus_alpha_div_alpha);
    }

    __shared__ float warp_sums[32];

    float block_sum = block_reduce_sum(val, warp_sums);

    if (threadIdx.x == 0)
    {
        atomicAdd(out, block_sum);
    }
}

/**
 * @brief Computes the specified raw moment of Levy process
 *
 * This kernel simulates multiple independent Levy process paths and computes
 * their raw moment of specified order at the end of the simulation.
 * The raw moment is calculated as E[X^r] where r is an integer.
 *
 * @param out Pointer to output value (accumulated sum of raw
 * moments, should be divided by particles count in host)
 * @param start_position Initial position for all particles
 * @param diffusivity Diffusion coefficient (D)
 * @param order Fractional order of the raw moment to compute
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param particles Number of particles to simulate
 * @param seed Random seed for CURAND
 */
extern "C" __global__ void raw_moment(float *out, float alpha, float start_position, int order,
                                      float duration, float time_step, float inv_alpha,
                                      float one_minus_alpha_div_alpha, size_t particles,
                                      unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 0.0f;

    if (idx < particles)
    {
        float end_position = end(alpha, start_position, duration, time_step, seed, idx,
                                 inv_alpha, one_minus_alpha_div_alpha);
        val = powf(end_position, order);
    }

    __shared__ float warp_sums[32];

    float block_sum = block_reduce_sum(val, warp_sums);

    if (threadIdx.x == 0)
    {
        atomicAdd(out, block_sum);
    }
}

/**
 * @brief Computes the specified fractional raw moment of Levy process
 *
 * This kernel simulates multiple independent Levy process paths and computes
 * their raw moment of specified fractional order at the end of the simulation.
 * The fractional raw moment is calculated as E[|X|^r] where r is a real number.
 *
 * @param out Pointer to output value (accumulated sum of fractional raw
 * moments, should be divided by particles count in host)
 * @param start_position Initial position for all particles
 * @param diffusivity Diffusion coefficient (D)
 * @param order Fractional order of the raw moment to compute
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param particles Number of particles to simulate
 * @param seed Random seed for CURAND
 */
extern "C" __global__ void
frac_raw_moment(float *out, float alpha, float start_position, float order,
                float duration, float time_step, float inv_alpha,
                float one_minus_alpha_div_alpha, size_t particles,
                unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 0.0f;

    if (idx < particles)
    {
        float end_position = end(alpha, start_position, duration, time_step, seed, idx,
                                 inv_alpha, one_minus_alpha_div_alpha);
        val = powf(fabsf(end_position), order);
    }

    __shared__ float warp_sums[32];

    float block_sum = block_reduce_sum(val, warp_sums);

    if (threadIdx.x == 0)
    {
        atomicAdd(out, block_sum);
    }
}

/**
 * @brief Computes the specified fractional central moment of Levy process
 *
 * This kernel simulates multiple independent Levy process paths and computes
 * their central moment of specified fractional order at the end of the
 * simulation. The fractional central moment is calculated as E[|X-μ|^r] where r
 * is a real number.
 *
 * @param out Pointer to output value (accumulated sum of fractional central
 * moments, should be divided by particles count in host)
 * @param mean Pre-computed mean of the process
 * @param order Fractional order of the central moment to compute
 * @param start_position Initial position for all particles
 * @param diffusivity Diffusion coefficient (D)
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param particles Number of particles to simulate
 * @param seed Random seed for CURAND
 */
extern "C" __global__ void
frac_central_moment(float *out, float mean, float alpha, float start_position,
                    float order, float duration, float time_step,
                    float inv_alpha, float one_minus_alpha_div_alpha,
                    size_t particles, unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 0.0f;

    if (idx < particles)
    {
        float end_position = end(alpha, start_position, duration, time_step, seed, idx,
                                 inv_alpha, one_minus_alpha_div_alpha);
        val = powf(fabsf(end_position - mean), order);
    }

    __shared__ float warp_sums[32];

    float block_sum = block_reduce_sum(val, warp_sums);

    if (threadIdx.x == 0)
    {
        atomicAdd(out, block_sum);
    }
}
