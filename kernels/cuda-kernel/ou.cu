/**
 * CUDA Kernel for Ornstein-Uhlenbeck Process Simulation
 *
 * The OU process is described by the SDE:
 * dX_t = -theta * X_t dt + sigma * dW_t
 *
 * where:
 * - theta: mean reversion speed
 * - sigma: volatility
 * - W_t: Wiener process
 */

#include <curand_kernel.h>
#include "utils.cu"

/**
 * Simulate Ornstein-Uhlenbeck process trajectories
 *
 * @param t Pointer to output array for time points
 * @param x Pointer to output array for particle positions
 * @param start_position Initial position for all particles
 * @param theta Mean reversion speed
 * @param sigma Volatility
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param seed Random seed for CURAND
 * @param idx Index of the particle
 */
QUALIFIERS void simulate(
    float *t,
    float *x,
    float start_position,
    float theta,
    float sigma,
    float duration,
    float time_step,
    unsigned long long seed,
    size_t idx)
{
    float current_x = start_position;
    float current_t = 0.0f;

    t[0] = current_t;
    x[0] = current_x;

    float scale = sqrtf(sigma * time_step);
    size_t num_steps = static_cast<size_t>(ceil(duration / time_step));

    curandState state;
    curand_init(seed, idx, 0, &state);

    float mu;
    float xi;

    for (size_t i = 0; i < num_steps - 1; ++i)
    {
        mu = -theta * current_x;
        xi = curand_normal(&state);
        current_x += mu * time_step + scale * xi;
        current_t += time_step;

        t[i + 1] = current_t;
        x[i + 1] = current_x;
    }

    float last_step = duration - current_t;
    xi = curand_normal(&state);
    mu = -theta * current_x;
    current_x += mu * last_step + sqrtf(sigma * last_step) * xi;

    t[num_steps] = duration;
    x[num_steps] = current_x;
}

/**
 * Simulate the end position of Ornstein-Uhlenbeck process trajectories
 *
 * @param start_position Initial position for all particles
 * @param theta Mean reversion speed
 * @param sigma Volatility
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param seed Random seed for CURAND
 * @param idx Index of the particle
 */
QUALIFIERS float end(
    float start_position,
    float theta,
    float sigma,
    float duration,
    float time_step,
    unsigned long long seed,
    size_t idx)
{
    float current_x = start_position;

    float scale = sqrtf(sigma * time_step);
    size_t num_steps = static_cast<size_t>(ceil(duration / time_step));

    curandState state;
    curand_init(seed, idx, 0, &state);

    float mu;
    float xi;

    for (size_t i = 0; i < num_steps - 1; ++i)
    {
        mu = -theta * current_x;
        xi = curand_normal(&state);
        current_x += mu * time_step + scale * xi;
    }

    float last_step = duration - (num_steps - 1) * time_step;
    xi = curand_normal(&state);
    mu = -theta * current_x;
    current_x += mu * last_step + sqrtf(sigma * last_step) * xi;

    return current_x - start_position;
}

/**
 * @brief Simulates Ornstein-Uhlenbeck process and computes the mean position across all particles
 *
 * This kernel simulates multiple independent Ornstein-Uhlenbeck process paths and computes
 * their mean position at the end of the simulation. Each thread handles one particle.
 *
 * @param out Pointer to output value (accumulated sum, should be divided by particles count in host)
 * @param start_position Initial position for all particles
 * @param theta Mean reversion speed
 * @param sigma Volatility
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param particles Number of particles to simulate
 * @param seed Random seed for CURAND
 */
extern "C" __global__ void mean(float *out,
                                float start_position, float theta,
                                float sigma,
                                float duration, float time_step,
                                size_t particles,
                                unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 0.0f;

    if (idx < particles)
    {
        val = end(start_position, theta, sigma, duration, time_step, seed, idx);
    }

    __shared__ float warp_sums[32];

    float block_sum = block_reduce_sum(val, warp_sums);

    if (threadIdx.x == 0)
    {
        atomicAdd(out, block_sum);
    }
}

/**
 * @brief Computes the mean squared displacement (MSD) of Ornstein-Uhlenbeck process
 *
 * This kernel simulates multiple independent Ornstein-Uhlenbeck process paths and computes
 * their mean squared displacement at the end of the simulation.
 *
 * @param out Pointer to output value (accumulated sum of squared displacements, should be divided by particles count in host)
 * @param start_position Initial position for all particles
 * @param theta Mean reversion speed
 * @param sigma Volatility
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param particles Number of particles to simulate
 * @param seed Random seed for CURAND
 */
extern "C" __global__ void msd(float *out, float start_position,
                               float theta, float sigma,
                               float duration, float time_step,
                               size_t particles,
                               unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 0.0f;

    if (idx < particles)
    {
        float end_position = end(start_position, theta, sigma, duration, time_step, seed, idx);
        val = (end_position - start_position) * (end_position - start_position);
    }

    __shared__ float warp_sums[32];

    float block_sum = block_reduce_sum(val, warp_sums);

    if (threadIdx.x == 0)
    {
        atomicAdd(out, block_sum);
    }
}

/**
 * @brief Computes the specified raw moment of Ornstein-Uhlenbeck process
 *
 * This kernel simulates multiple independent Ornstein-Uhlenbeck process paths and computes
 * their raw moment of specified integer order at the end of the simulation.
 * The raw moment is calculated as E[X^n].
 *
 * @param out Pointer to output value (accumulated sum of raw moments, should be divided by particles count in host)
 * @param start_position Initial position for all particles
 * @param theta Mean reversion speed
 * @param sigma Volatility
 * @param order Integer order of the raw moment to compute
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param particles Number of particles to simulate
 * @param seed Random seed for CURAND
 */
extern "C" __global__ void raw_moment(float *out,
                                      float start_position,
                                      float theta, float sigma, int order,
                                      float duration, float time_step,
                                      size_t particles,
                                      unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 0.0f;

    if (idx < particles)
    {
        float end_position = end(start_position, theta, sigma, duration, time_step, seed, idx);
        val = pow(end_position, order);
    }

    __shared__ float warp_sums[32];

    float block_sum = block_reduce_sum(val, warp_sums);

    if (threadIdx.x == 0)
    {
        atomicAdd(out, block_sum);
    }
}

/**
 * @brief Computes the specified central moment of Ornstein-Uhlenbeck process
 *
 * This kernel simulates multiple independent Ornstein-Uhlenbeck process paths and computes
 * their central moment of specified integer order at the end of the simulation.
 * The central moment is calculated as E[(X-μ)^n] where μ is the mean.
 *
 * @param out Pointer to output value (accumulated sum of central moments, should be divided by particles count in host)
 * @param order Integer order of the central moment to compute
 * @param mean Pre-computed mean of the process
 * @param start_position Initial position for all particles
 * @param theta Mean reversion speed
 * @param sigma Volatility
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param particles Number of particles to simulate
 * @param seed Random seed for CURAND
 */
extern "C" __global__ void central_moment(float *out,
                                          int order, float mean, float start_position,
                                          float theta, float sigma,
                                          float duration, float time_step,
                                          size_t particles,
                                          unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 0.0f;

    if (idx < particles)
    {
        float end_position = end(start_position, theta, sigma, duration, time_step, seed, idx);
        val = pow(end_position - mean, order);
    }

    __shared__ float warp_sums[32];

    float block_sum = block_reduce_sum(val, warp_sums);

    if (threadIdx.x == 0)
    {
        atomicAdd(out, block_sum);
    }
}

/**
 * @brief Computes the specified fractional raw moment of Ornstein-Uhlenbeck process
 *
 * This kernel simulates multiple independent Ornstein-Uhlenbeck process paths and computes
 * their raw moment of specified fractional order at the end of the simulation.
 * The fractional raw moment is calculated as E[|X|^r] where r is a real number.
 *
 * @param out Pointer to output value (accumulated sum of fractional raw moments, should be divided by particles count in host)
 * @param start_position Initial position for all particles
 * @param theta Mean reversion speed
 * @param sigma Volatility
 * @param order Fractional order of the raw moment to compute
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param particles Number of particles to simulate
 * @param seed Random seed for CURAND
 */
extern "C" __global__ void frac_raw_moment(float *out,
                                           float start_position,
                                           float theta, float sigma, float order,
                                           float duration, float time_step,
                                           size_t particles,
                                           unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 0.0f;

    if (idx < particles)
    {
        float end_position = end(start_position, theta, sigma, duration, time_step, seed, idx);
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
 * @brief Computes the specified fractional central moment of Ornstein-Uhlenbeck process
 *
 * This kernel simulates multiple independent Ornstein-Uhlenbeck process paths and computes
 * their central moment of specified fractional order at the end of the simulation.
 * The fractional central moment is calculated as E[|X-μ|^r] where r is a real number.
 *
 * @param out Pointer to output value (accumulated sum of fractional central moments, should be divided by particles count in host)
 * @param mean Pre-computed mean of the process
 * @param order Fractional order of the central moment to compute
 * @param start_position Initial position for all particles
 * @param theta Mean reversion speed
 * @param sigma Volatility
 * @param theta Mean reversion speed
 * @param sigma Volatility
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param particles Number of particles to simulate
 * @param seed Random seed for CURAND
 */
extern "C" __global__ void frac_central_moment(float *out,
                                               float mean,
                                               float order,
                                               float start_position,
                                               float theta, float sigma,
                                               float duration, float time_step,
                                               size_t particles,
                                               unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 0.0f;

    if (idx < particles)
    {
        float end_position = end(start_position, theta, sigma, duration, time_step, seed, idx);
        val = powf(fabsf(end_position - mean), order);
    }

    __shared__ float warp_sums[32];

    float block_sum = block_reduce_sum(val, warp_sums);

    if (threadIdx.x == 0)
    {
        atomicAdd(out, block_sum);
    }
}
