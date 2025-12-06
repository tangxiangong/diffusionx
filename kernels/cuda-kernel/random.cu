#include <curand_kernel.h>
#include <math_constants.h>

extern "C" __global__ void randexp(float *out, size_t len, unsigned long long seed)
{
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < len)
    {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float u = curand_uniform(&state);
        out[idx] = -logf(u);
    }
}

QUALIFIERS float sample_standard_alpha_one(float alpha, float beta,
                                           curandStatePhilox4_32_10_t *state)
{
    float v = curand_uniform(state) * CUDART_PI_F - CUDART_PIO2_F;
    float w = -logf(curand_uniform(state));
    float half_pi_plus_beta_v = CUDART_PIO2_F + beta * v;
    float c1 = half_pi_plus_beta_v * tanf(v);
    float c2 = (CUDART_PIO2_F * w * cosf(v)) / logf(half_pi_plus_beta_v) * beta;
    return (c1 - c2) * CUDART_2_OVER_PI_F;
}

QUALIFIERS float sample_standard_alpha_with_constants(
    float alpha, float inv_alpha, float one_minus_alpha_div_alpha, float b,
    float s, curandStatePhilox4_32_10_t *state)
{
    float v = curand_uniform(state) * CUDART_PI_F - CUDART_PIO2_F;
    float w = -logf(curand_uniform(state));
    float v_plus_b = v + b;
    float cos_v = cosf(v);
    float c1 = alpha * sinf(v_plus_b) / powf(cos_v, inv_alpha);
    float c2 = powf(cosf(v - alpha * v_plus_b) / w, one_minus_alpha_div_alpha);
    return s * c1 * c2;
}

extern "C" __global__ void
standard_stable_rand(float *out, float alpha, float beta,
                     float inv_alpha,
                     float one_minus_alpha_div_alpha, float b, float s, size_t len,
                     unsigned long long seed)
{
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < len)
    {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);
        float r;
        if (abs(alpha - 1.0f) < 1e-3)
        {
            r = sample_standard_alpha_one(alpha, beta, &state);
        }
        else
        {
            r = sample_standard_alpha_with_constants(
                alpha, inv_alpha, one_minus_alpha_div_alpha, b, s, &state);
        }
        out[idx] = r;
    }
}