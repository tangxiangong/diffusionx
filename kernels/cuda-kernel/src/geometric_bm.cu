/**
 * CUDA Kernel for Geometric Brownian Motion (geometric_bm.rs)
 * 
 * Corresponds to: src/simulation/continuous/geometric_bm.rs
 * 
 * SDE: dS_t = μ S_t dt + σ S_t dW_t
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C" {

/**
 * Simulate Geometric Brownian Motion trajectories (single precision)
 * 
 * @param states cuRAND states for random number generation
 * @param positions Output array [num_particles * (num_steps + 1)]
 * @param start_position Initial position S_0
 * @param mu Drift coefficient (μ)
 * @param sigma Volatility (σ)
 * @param time_step Time step (dt)
 * @param num_steps Number of time steps
 * @param num_particles Number of particles to simulate
 */
__global__ void simulate_geometric_bm_f32(
    curandState* states,
    float* positions,
    float start_position,
    float mu,
    float sigma,
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
    
    float sqrt_dt = sqrtf(time_step);
    
    for (int step = 0; step < num_steps; step++) {
        float random_normal = curand_normal(&local_state);
        
        // GBM exact solution: S(t+dt) = S(t) * exp((μ - σ²/2)dt + σ√dt Z)
        float drift = (mu - 0.5f * sigma * sigma) * time_step;
        float diffusion = sigma * sqrt_dt * random_normal;
        position *= expf(drift + diffusion);
        
        positions[offset + step + 1] = position;
    }
    
    states[idx] = local_state;
}

/**
 * Simulate Geometric Brownian Motion trajectories (double precision)
 */
__global__ void simulate_geometric_bm_f64(
    curandState* states,
    double* positions,
    double start_position,
    double mu,
    double sigma,
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
    
    double sqrt_dt = sqrt(time_step);
    
    for (int step = 0; step < num_steps; step++) {
        double random_normal = curand_normal_double(&local_state);
        
        double drift = (mu - 0.5 * sigma * sigma) * time_step;
        double diffusion = sigma * sqrt_dt * random_normal;
        position *= exp(drift + diffusion);
        
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