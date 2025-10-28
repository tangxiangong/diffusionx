/**
 * CUDA Kernel for Fractional Brownian Motion (fbm.rs)
 * 
 * Corresponds to: src/simulation/continuous/fbm.rs
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C" {

__global__ void simulate_fbm_f32(
    curandState* states,
    float* positions,
    float* covariance_matrix,  // Pre-computed covariance
    float hurst,
    float time_step,
    int num_steps,
    int num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    curandState local_state = states[idx];
    int offset = idx * (num_steps + 1);
    
    positions[offset] = 0.0f;
    
    // Generate standard normal increments
    float* increments = new float[num_steps];
    for (int i = 0; i < num_steps; i++) {
        increments[i] = curand_normal(&local_state);
    }
    
    // Apply covariance structure (Cholesky decomposition)
    float position = 0.0f;
    for (int step = 0; step < num_steps; step++) {
        float increment = 0.0f;
        for (int j = 0; j <= step; j++) {
            increment += covariance_matrix[step * num_steps + j] * increments[j];
        }
        position += increment;
        positions[offset + step + 1] = position;
    }
    
    delete[] increments;
    states[idx] = local_state;
}

__global__ void init_curand_states(
    curandState* states,
    unsigned long long seed,
    int num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    curand_init(seed, idx, 0, &states[idx]);
}

}
