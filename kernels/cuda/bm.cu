/**
 * CUDA Kernel for Brownian Motion (bm.rs)
 * 
 * Corresponds to: src/simulation/continuous/bm.rs
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C" {

/**
 * Simulate Brownian motion trajectories (single precision)
 * 
 * @param states cuRAND states for random number generation
 * @param positions Output array [num_particles * (num_steps + 1)]
 * @param start_position Initial position for all particles
 * @param diffusion_coefficient Diffusion coefficient (D)
 * @param time_step Time step (dt)
 * @param num_steps Number of time steps
 * @param num_particles Number of particles to simulate
 */
__global__ void simulate_bm_f32(
    curandState* states,
    float* positions,
    float start_position,
    float diffusion_coefficient,
    float time_step,
    int num_steps,
    int num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    curandState local_state = states[idx];
    float position = start_position;
    int offset = idx * (num_steps + 1);
    
    positions[offset] = position;
    
    // Precompute noise scaling: sqrt(2 * D * dt)
    float noise_scale = sqrtf(2.0f * diffusion_coefficient * time_step);
    
    for (int step = 0; step < num_steps; step++) {
        float random_normal = curand_normal(&local_state);
        position += noise_scale * random_normal;
        positions[offset + step + 1] = position;
    }
    
    states[idx] = local_state;
}

/**
 * Simulate Brownian motion trajectories (double precision)
 */
__global__ void simulate_bm_f64(
    curandState* states,
    double* positions,
    double start_position,
    double diffusion_coefficient,
    double time_step,
    int num_steps,
    int num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    curandState local_state = states[idx];
    double position = start_position;
    int offset = idx * (num_steps + 1);
    
    positions[offset] = position;
    
    double noise_scale = sqrt(2.0 * diffusion_coefficient * time_step);
    
    for (int step = 0; step < num_steps; step++) {
        double random_normal = curand_normal_double(&local_state);
        position += noise_scale * random_normal;
        positions[offset + step + 1] = position;
    }
    
    states[idx] = local_state;
}

/**
 * Initialize cuRAND states for each thread
 */
__global__ void init_curand_states(
    curandState* states,
    unsigned long long seed,
    int num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    curand_init(seed, idx, 0, &states[idx]);
}

} // extern "C"