/**
 * CUDA Kernel for Ornstein-Uhlenbeck Process Simulation
 * 
 * The OU process is described by the SDE:
 * dX_t = theta * (mu - X_t) dt + sigma * dW_t
 * 
 * where:
 * - theta: mean reversion speed
 * - mu: long-term mean
 * - sigma: volatility
 * - W_t: Wiener process
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C" {

/**
 * Simulate Ornstein-Uhlenbeck process trajectories (single precision)
 * 
 * @param states cuRAND states for random number generation
 * @param positions Output array for particle positions [num_particles * (num_steps + 1)]
 * @param start_position Initial position for all particles
 * @param theta Mean reversion speed
 * @param mu Long-term mean
 * @param sigma Volatility
 * @param time_step Time step (dt)
 * @param num_steps Number of time steps
 * @param num_particles Number of particles to simulate
 */
__global__ void simulate_ou_process_f32(
    curandState* states,
    float* positions,
    float start_position,
    float theta,
    float mu,
    float sigma,
    float time_step,
    int num_steps,
    int num_particles
) {
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_idx >= num_particles) return;
    
    curandState local_state = states[particle_idx];
    
    float position = start_position;
    int offset = particle_idx * (num_steps + 1);
    
    // Store initial position
    positions[offset] = position;
    
    float sqrt_dt = sqrtf(time_step);
    
    // Simulate trajectory using Euler-Maruyama method
    for (int step = 0; step < num_steps; step++) {
        float random_normal = curand_normal(&local_state);
        
        // Drift term: theta * (mu - X_t) * dt
        float drift = theta * (mu - position) * time_step;
        
        // Diffusion term: sigma * sqrt(dt) * N(0,1)
        float diffusion = sigma * sqrt_dt * random_normal;
        
        position += drift + diffusion;
        positions[offset + step + 1] = position;
    }
    
    states[particle_idx] = local_state;
}

/**
 * Simulate Ornstein-Uhlenbeck process trajectories (double precision)
 */
__global__ void simulate_ou_process_f64(
    curandState* states,
    double* positions,
    double start_position,
    double theta,
    double mu,
    double sigma,
    double time_step,
    int num_steps,
    int num_particles
) {
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_idx >= num_particles) return;
    
    curandState local_state = states[particle_idx];
    
    double position = start_position;
    int offset = particle_idx * (num_steps + 1);
    
    positions[offset] = position;
    
    double sqrt_dt = sqrt(time_step);
    
    for (int step = 0; step < num_steps; step++) {
        double random_normal = curand_normal_double(&local_state);
        
        double drift = theta * (mu - position) * time_step;
        double diffusion = sigma * sqrt_dt * random_normal;
        
        position += drift + diffusion;
        positions[offset + step + 1] = position;
    }
    
    states[particle_idx] = local_state;
}

/**
 * Simulate OU process with analytical solution (more accurate)
 * 
 * Uses exact solution:
 * X(t+dt) = X(t) * exp(-theta*dt) + mu * (1 - exp(-theta*dt)) + sigma * sqrt((1-exp(-2*theta*dt))/(2*theta)) * N(0,1)
 */
__global__ void simulate_ou_process_exact_f32(
    curandState* states,
    float* positions,
    float start_position,
    float theta,
    float mu,
    float sigma,
    float time_step,
    int num_steps,
    int num_particles
) {
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_idx >= num_particles) return;
    
    curandState local_state = states[particle_idx];
    
    float position = start_position;
    int offset = particle_idx * (num_steps + 1);
    
    positions[offset] = position;
    
    // Precompute constants
    float exp_neg_theta_dt = expf(-theta * time_step);
    float one_minus_exp = 1.0f - exp_neg_theta_dt;
    float noise_scale = sigma * sqrtf((1.0f - expf(-2.0f * theta * time_step)) / (2.0f * theta));
    
    for (int step = 0; step < num_steps; step++) {
        float random_normal = curand_normal(&local_state);
        
        // Exact solution
        position = position * exp_neg_theta_dt + mu * one_minus_exp + noise_scale * random_normal;
        
        positions[offset + step + 1] = position;
    }
    
    states[particle_idx] = local_state;
}

/**
 * Simulate OU process with analytical solution (double precision)
 */
__global__ void simulate_ou_process_exact_f64(
    curandState* states,
    double* positions,
    double start_position,
    double theta,
    double mu,
    double sigma,
    double time_step,
    int num_steps,
    int num_particles
) {
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_idx >= num_particles) return;
    
    curandState local_state = states[particle_idx];
    
    double position = start_position;
    int offset = particle_idx * (num_steps + 1);
    
    positions[offset] = position;
    
    double exp_neg_theta_dt = exp(-theta * time_step);
    double one_minus_exp = 1.0 - exp_neg_theta_dt;
    double noise_scale = sigma * sqrt((1.0 - exp(-2.0 * theta * time_step)) / (2.0 * theta));
    
    for (int step = 0; step < num_steps; step++) {
        double random_normal = curand_normal_double(&local_state);
        
        position = position * exp_neg_theta_dt + mu * one_minus_exp + noise_scale * random_normal;
        
        positions[offset + step + 1] = position;
    }
    
    states[particle_idx] = local_state;
}

} // extern "C"