#include <curand_kernel.h>

extern "C" __global__ void bm_mean(float *out, float start_position,
                                   float diffusivity, float duration,
                                   float time_step, size_t particles,
                                   unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float current_x = start_position;
    float sigma = sqrtf(2.0f * diffusivity * time_step);
    size_t num_steps = static_cast<size_t>(ceil(duration / time_step));
    if (idx < particles)
    {
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
    }

    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    sdata[tid] = current_x;
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

extern "C" __global__ void bm_msd(float *out, float diffusivity, float duration,
                                  float time_step, size_t particles,
                                  unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float current_x = 0.0;
    float sd;
    float sigma = sqrtf(2.0f * diffusivity * time_step);
    size_t num_steps = static_cast<size_t>(ceil(duration / time_step));
    if (idx < particles)
    {
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
        sd = current_x * current_x;
    }

    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    sdata[tid] = sd;
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

extern "C" __global__ void
bm_central_moment(float *out, float mean, float start_position,
                  float diffusivity, int order, float duration, float time_step,
                  size_t particles, unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float current_x = start_position;
    float sd;
    float sigma = sqrtf(2.0f * diffusivity * time_step);
    size_t num_steps = static_cast<size_t>(ceil(duration / time_step));
    if (idx < particles)
    {
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
        sd = pow(current_x - mean, order);
    }

    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    sdata[tid] = sd;
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

extern "C" __global__ void
bm_raw_moment(float *out, float start_position,
              float diffusivity, int order, float duration, float time_step,
              size_t particles, unsigned long long seed)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float current_x = start_position;
    float sd;
    float sigma = sqrtf(2.0f * diffusivity * time_step);
    size_t num_steps = static_cast<size_t>(ceil(duration / time_step));
    if (idx < particles)
    {
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
        sd = pow(current_x, order);
    }

    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    sdata[tid] = sd;
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
