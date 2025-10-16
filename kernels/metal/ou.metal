#include <metal_stdlib>
using namespace metal;

/**
 * Metal Shader for Ornstein-Uhlenbeck Process Simulation
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

/**
 * Simulate Ornstein-Uhlenbeck process trajectories using Euler-Maruyama method
 *
 * @param random_normals Pre-generated normal random numbers [num_particles * num_steps]
 * @param positions Output array for particle positions [num_particles * (num_steps + 1)]
 * @param start_position Initial position for all particles
 * @param theta Mean reversion speed
 * @param mu Long-term mean
 * @param sigma Volatility
 * @param time_step Time step (dt)
 * @param num_steps Number of time steps
 * @param num_particles Number of particles to simulate
 * @param particle_idx Thread index (particle ID)
 */
kernel void simulate_ou_process_euler(
    device const float* random_normals [[buffer(0)]],
    device float* positions [[buffer(1)]],
    constant float& start_position [[buffer(2)]],
    constant float& theta [[buffer(3)]],
    constant float& mu [[buffer(4)]],
    constant float& sigma [[buffer(5)]],
    constant float& time_step [[buffer(6)]],
    constant int& num_steps [[buffer(7)]],
    constant int& num_particles [[buffer(8)]],
    uint particle_idx [[thread_position_in_grid]]
) {
    if (particle_idx >= uint(num_particles)) return;

    float position = start_position;
    int offset = particle_idx * (num_steps + 1);
    int random_offset = particle_idx * num_steps;

    // Store initial position
    positions[offset] = position;

    float sqrt_dt = sqrt(time_step);

    // Simulate trajectory using Euler-Maruyama method
    for (int step = 0; step < num_steps; step++) {
        float random_normal = random_normals[random_offset + step];

        // Drift term: theta * (mu - X_t) * dt
        float drift = theta * (mu - position) * time_step;

        // Diffusion term: sigma * sqrt(dt) * N(0,1)
        float diffusion = sigma * sqrt_dt * random_normal;

        position += drift + diffusion;
        positions[offset + step + 1] = position;
    }
}

/**
 * Simulate OU process using exact solution (more accurate)
 *
 * Uses the analytical solution:
 * X(t+dt) = X(t) * exp(-theta*dt) + mu * (1 - exp(-theta*dt))
 *           + sigma * sqrt((1-exp(-2*theta*dt))/(2*theta)) * N(0,1)
 *
 * This method is numerically more stable and accurate than Euler-Maruyama.
 */
kernel void simulate_ou_process_exact(
    device const float* random_normals [[buffer(0)]],
    device float* positions [[buffer(1)]],
    constant float& start_position [[buffer(2)]],
    constant float& theta [[buffer(3)]],
    constant float& mu [[buffer(4)]],
    constant float& sigma [[buffer(5)]],
    constant float& time_step [[buffer(6)]],
    constant int& num_steps [[buffer(7)]],
    constant int& num_particles [[buffer(8)]],
    uint particle_idx [[thread_position_in_grid]]
) {
    if (particle_idx >= uint(num_particles)) return;

    float position = start_position;
    int offset = particle_idx * (num_steps + 1);
    int random_offset = particle_idx * num_steps;

    positions[offset] = position;

    // Precompute constants
    float exp_neg_theta_dt = exp(-theta * time_step);
    float one_minus_exp = 1.0f - exp_neg_theta_dt;
    float noise_scale = sigma * sqrt((1.0f - exp(-2.0f * theta * time_step)) / (2.0f * theta));

    for (int step = 0; step < num_steps; step++) {
        float random_normal = random_normals[random_offset + step];

        // Exact solution
        position = position * exp_neg_theta_dt + mu * one_minus_exp + noise_scale * random_normal;

        positions[offset + step + 1] = position;
    }
}

/**
 * Simulate OU process with time-varying parameters
 * Allows theta, mu, and sigma to change at each time step
 */
kernel void simulate_ou_process_time_varying(
    device const float* random_normals [[buffer(0)]],
    device const float* theta_array [[buffer(1)]],
    device const float* mu_array [[buffer(2)]],
    device const float* sigma_array [[buffer(3)]],
    device float* positions [[buffer(4)]],
    constant float& start_position [[buffer(5)]],
    constant float& time_step [[buffer(6)]],
    constant int& num_steps [[buffer(7)]],
    constant int& num_particles [[buffer(8)]],
    uint particle_idx [[thread_position_in_grid]]
) {
    if (particle_idx >= uint(num_particles)) return;

    float position = start_position;
    int offset = particle_idx * (num_steps + 1);
    int random_offset = particle_idx * num_steps;

    positions[offset] = position;

    float sqrt_dt = sqrt(time_step);

    for (int step = 0; step < num_steps; step++) {
        float theta = theta_array[step];
        float mu = mu_array[step];
        float sigma = sigma_array[step];
        float random_normal = random_normals[random_offset + step];

        float drift = theta * (mu - position) * time_step;
        float diffusion = sigma * sqrt_dt * random_normal;

        position += drift + diffusion;
        positions[offset + step + 1] = position;
    }
}

/**
 * Compute stationary variance of OU process
 *
 * For OU process, the stationary variance is: sigma^2 / (2 * theta)
 * This kernel computes the variance from trajectories for comparison
 */
kernel void compute_ou_variance(
    device const float* positions [[buffer(0)]],
    device float* variances [[buffer(1)]],
    device const float* means [[buffer(2)]],
    constant int& num_particles [[buffer(3)]],
    constant int& num_steps [[buffer(4)]],
    uint step_idx [[thread_position_in_grid]]
) {
    if (step_idx > uint(num_steps)) return;

    float mean = means[step_idx];
    float sum_sq_dev = 0.0f;

    for (int i = 0; i < num_particles; i++) {
        float pos = positions[i * (num_steps + 1) + step_idx];
        float deviation = pos - mean;
        sum_sq_dev += deviation * deviation;
    }

    variances[step_idx] = sum_sq_dev / float(num_particles);
}

/**
 * Compute mean reversion speed empirically
 * Useful for parameter estimation
 */
kernel void estimate_mean_reversion(
    device const float* positions [[buffer(0)]],
    device float* theta_estimates [[buffer(1)]],
    constant float& mu [[buffer(2)]],
    constant float& time_step [[buffer(3)]],
    constant int& num_particles [[buffer(4)]],
    constant int& num_steps [[buffer(5)]],
    uint particle_idx [[thread_position_in_grid]]
) {
    if (particle_idx >= uint(num_particles)) return;

    int offset = particle_idx * (num_steps + 1);
    float sum_theta = 0.0f;
    int count = 0;

    for (int step = 0; step < num_steps; step++) {
        float x_t = positions[offset + step];
        float x_t_plus_dt = positions[offset + step + 1];

        float dx = x_t_plus_dt - x_t;
        float deviation = x_t - mu;

        // Avoid division by zero
        if (abs(deviation) > 1e-6f) {
            float theta_estimate = -dx / (deviation * time_step);
            if (theta_estimate > 0.0f && theta_estimate < 100.0f) {
                sum_theta += theta_estimate;
                count++;
            }
        }
    }

    theta_estimates[particle_idx] = (count > 0) ? (sum_theta / float(count)) : 0.0f;
}
