/**
 * CUDA Kernel for Brownian Motion (bm.rs)
 *
 * Corresponds to: src/simulation/continuous/bm.rs and src/gpu/bm.rs
 *
 * This file implements various CUDA kernels for computing statistical properties
 * of Brownian motion processes through Monte Carlo simulation.
 */

#include <curand_kernel.h>

/**
 * @brief Simulates the Brownian motion for a single particle
 *
 * @param t Pointer to output array for time points
 * @param x Pointer to output array for particle positions
 * @param start_position Initial position of the particle
 * @param diffusivity Diffusion coefficient (D) controlling the rate of diffusion
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param seed Random seed for CURAND
 * @param idx Index of the particle
 */
QUALIFIERS void simulate(float *t, float *x, float start_position, float diffusivity, float duration, float time_step, unsigned long long seed, size_t idx)
{
    float current_x = start_position;
    float current_t = 0.0f;

    t[0] = current_t;
    x[0] = current_x;

    float sigma = sqrtf(2.0f * diffusivity * time_step);
    size_t num_steps = static_cast<size_t>(ceil(duration / time_step));

    curandState state;
    curand_init(seed, idx, 0, &state);

    for (size_t i = 0; i < num_steps - 1; ++i)
    {
        float xi = curand_normal(&state);
        current_x += xi * sigma;
        current_t += time_step;
        t[i + 1] = current_t;
        x[i + 1] = current_x;
    }

    float last_step = duration - current_t;
    float xi = curand_normal(&state);
    current_x += xi * sqrtf(2.0f * diffusivity * last_step);

    t[num_steps] = duration;
    x[num_steps] = current_x;
}

/**
 * @brief Simulates the displacement of Brownian motion for a single particle
 *
 * @param start_position Initial position of the particle
 * @param diffusivity Diffusion coefficient (D) controlling the rate of diffusion
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param seed Random seed for CURAND
 * @param idx Index of the particle
 */
QUALIFIERS float displacement(float start_position, float diffusivity, float duration, float time_step, unsigned long long seed, size_t idx)
{
    float current_x = start_position;

    float sigma = sqrtf(2.0f * diffusivity * time_step);
    size_t num_steps = static_cast<size_t>(ceil(duration / time_step));

    curandState state;
    curand_init(seed, idx, 0, &state);

    for (size_t i = 0; i < num_steps - 1; ++i)
    {
        float xi = curand_normal(&state);
        current_x += xi * sigma;
    }

    float last_step = duration - (num_steps - 1) * time_step;
    float xi = curand_normal(&state);
    current_x += xi * sqrtf(2.0f * diffusivity * last_step);
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
 * @param seed Random seed for CURAND
 */
extern "C" __global__ void mean(float *out,
                                float start_position, float diffusivity,
                                float duration, float time_step,
                                size_t particles,
                                unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 0.0f;

    if (idx < particles)
    {
        val = displacement(start_position, diffusivity, duration, time_step, seed, idx);
    }

    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(out, sdata[0]);
    }
}

/**
 * @brief Computes the mean squared displacement (MSD) of Brownian motion
 *
 * This kernel simulates multiple independent Brownian motion paths and computes
 * their mean squared displacement at the end of the simulation.
 *
 * @param out Pointer to output value (accumulated sum of squared displacements, should be divided by particles count in host)
 * @param diffusivity Diffusion coefficient (D)
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param particles Number of particles to simulate
 * @param seed Random seed for CURAND
 */
extern "C" __global__ void msd(float *out, float start_position,
                               float diffusivity, float duration,
                               float time_step, size_t particles,
                               unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 0.0f;

    if (idx < particles)
    {
        float end = displacement(0.0f, diffusivity, duration, time_step, seed, idx);
        val = (end - start_position) * (end - start_position);
    }

    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(out, sdata[0]);
    }
}

/**
 * @brief Computes the specified raw moment of Brownian motion
 *
 * This kernel simulates multiple independent Brownian motion paths and computes
 * their raw moment of specified integer order at the end of the simulation.
 * The raw moment is calculated as E[X^n].
 *
 * @param out Pointer to output value (accumulated sum of raw moments, should be divided by particles count in host)
 * @param start_position Initial position for all particles
 * @param diffusivity Diffusion coefficient (D)
 * @param order Integer order of the raw moment to compute
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param particles Number of particles to simulate
 * @param seed Random seed for CURAND
 */
extern "C" __global__ void raw_moment(float *out,
                                      float start_position,
                                      float diffusivity, int order,
                                      float duration, float time_step,
                                      size_t particles,
                                      unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 0.0f;

    if (idx < particles)
    {
        float end = displacement(start_position, diffusivity, duration, time_step, seed, idx);
        val = pow(end, order);
    }

    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(out, sdata[0]);
    }
}

/**
 * @brief Computes the specified central moment of Brownian motion
 *
 * This kernel simulates multiple independent Brownian motion paths and computes
 * their central moment of specified integer order at the end of the simulation.
 * The central moment is calculated as E[(X-μ)^n] where μ is the mean.
 *
 * @param out Pointer to output value (accumulated sum of central moments, should be divided by particles count in host)
 * @param order Integer order of the central moment to compute
 * @param mean Pre-computed mean of the process
 * @param start_position Initial position for all particles
 * @param diffusivity Diffusion coefficient (D)
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param particles Number of particles to simulate
 * @param seed Random seed for CURAND
 */
extern "C" __global__ void central_moment(float *out,
                                          int order, float mean, float start_position,
                                          float diffusivity,
                                          float duration, float time_step,
                                          size_t particles,
                                          unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 0.0f;

    if (idx < particles)
    {
        float end = displacement(start_position, diffusivity, duration, time_step, seed, idx);
        val = pow(end - mean, order);
    }

    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(out, sdata[0]);
    }
}

/**
 * @brief Computes the specified fractional raw moment of Brownian motion
 *
 * This kernel simulates multiple independent Brownian motion paths and computes
 * their raw moment of specified fractional order at the end of the simulation.
 * The fractional raw moment is calculated as E[|X|^r] where r is a real number.
 *
 * @param out Pointer to output value (accumulated sum of fractional raw moments, should be divided by particles count in host)
 * @param start_position Initial position for all particles
 * @param diffusivity Diffusion coefficient (D)
 * @param order Fractional order of the raw moment to compute
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param particles Number of particles to simulate
 * @param seed Random seed for CURAND
 */
extern "C" __global__ void frac_raw_moment(float *out,
                                           float start_position,
                                           float diffusivity, float order,
                                           float duration, float time_step,
                                           size_t particles,
                                           unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 0.0f;

    if (idx < particles)
    {
        float end = displacement(start_position, diffusivity, duration, time_step, seed, idx);
        val = powf(fabsf(end), order);
    }

    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(out, sdata[0]);
    }
}

/**
 * @brief Computes the specified fractional central moment of Brownian motion
 *
 * This kernel simulates multiple independent Brownian motion paths and computes
 * their central moment of specified fractional order at the end of the simulation.
 * The fractional central moment is calculated as E[|X-μ|^r] where r is a real number.
 *
 * @param out Pointer to output value (accumulated sum of fractional central moments, should be divided by particles count in host)
 * @param mean Pre-computed mean of the process
 * @param order Fractional order of the central moment to compute
 * @param start_position Initial position for all particles
 * @param diffusivity Diffusion coefficient (D)
 * @param duration Total simulation time
 * @param time_step Time step size for the simulation
 * @param particles Number of particles to simulate
 * @param seed Random seed for CURAND
 */
extern "C" __global__ void frac_central_moment(float *out,
                                               float mean,
                                               float order,
                                               float start_position,
                                               float diffusivity,
                                               float duration, float time_step,
                                               size_t particles,
                                               unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 0.0f;

    if (idx < particles)
    {
        float end = displacement(start_position, diffusivity, duration, time_step, seed, idx);
        val = powf(fabsf(end - mean), order);
    }

    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(out, sdata[0]);
    }
}
