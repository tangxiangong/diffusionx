#include <curand_kernel.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Stable 分布采样（Chambers-Mallows-Stuck 方法）
// alpha: 稳定性参数 (0, 2]
// beta: 偏度参数 [-1, 1]
extern "C" __global__ void generate_stable_f32(curandState *states,
                                               float *output, float alpha,
                                               float beta, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  curandState local_state = states[idx];

  // 生成均匀分布 U ~ Uniform(-π/2, π/2)
  float u = curand_uniform(&local_state) * M_PI - M_PI / 2.0f;

  // 生成指数分布 W ~ Exp(1)
  float w = -logf(curand_uniform(&local_state));

  float result;

  // alpha = 1 的特殊情况
  if (fabsf(alpha - 1.0f) < 1e-6f) {
    float xi = M_PI / 2.0f + beta * u;
    result = (2.0f / M_PI) * (xi * tanf(u) - beta * logf(w));
  } else {
    // 一般情况
    float zeta = -beta * tanf(M_PI * alpha / 2.0f);
    float xi = atanf(zeta) / alpha;

    float part1 = sinf(alpha * (u + xi));
    float part2 = powf(cosf(u), 1.0f / alpha);
    float part3 = cosf((1.0f - alpha) * (u + xi)) / w;
    float part3_pow = powf(part3, (1.0f - alpha) / alpha);

    result = part1 / part2 * part3_pow;
  }

  output[idx] = result;
  states[idx] = local_state;
}

// 双精度版本
extern "C" __global__ void generate_stable_f64(curandState *states,
                                               double *output, double alpha,
                                               double beta, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  curandState local_state = states[idx];

  double u = curand_uniform_double(&local_state) * M_PI - M_PI / 2.0;
  double w = -log(curand_uniform_double(&local_state));

  double result;

  if (fabs(alpha - 1.0) < 1e-10) {
    double xi = M_PI / 2.0 + beta * u;
    result = (2.0 / M_PI) * (xi * tan(u) - beta * log(w));
  } else {
    double zeta = -beta * tan(M_PI * alpha / 2.0);
    double xi = atan(zeta) / alpha;

    double part1 = sin(alpha * (u + xi));
    double part2 = pow(cos(u), 1.0 / alpha);
    double part3 = cos((1.0 - alpha) * (u + xi)) / w;
    double part3_pow = pow(part3, (1.0 - alpha) / alpha);

    result = part1 / part2 * part3_pow;
  }

  output[idx] = result;
  states[idx] = local_state;
}

// 初始化 cuRAND 状态
extern "C" __global__ void
init_curand_states_stable(curandState *states, unsigned long long seed, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  curand_init(seed, idx, 0, &states[idx]);
}
