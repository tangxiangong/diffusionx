/**
 * CUDA Kernel for Monte Carlo Statistics Computation
 * 
 * This kernel computes statistical quantities (mean, MSD, variance) 
 * directly on GPU from simulated particle trajectories.
 */

#include <cuda_runtime.h>

extern "C" {

/**
 * Compute mean, MSD, and variance for Monte Carlo simulation
 * 
 * This kernel processes all time steps in parallel, computing statistics
 * across all particles at each time point.
 * 
 * @param positions Input array [num_particles * (num_steps + 1)]
 * @param mean Output array for mean [num_steps + 1]
 * @param msd Output array for mean square displacement [num_steps + 1]
 * @param variance Output array for variance [num_steps + 1]
 * @param num_particles Number of particles
 * @param num_steps Number of time steps
 */
__global__ void compute_stats_f32(
    const float* positions,
    float* mean,
    float* msd,
    float* variance,
    int num_particles,
    int num_steps
) {
    int step_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (step_idx > num_steps) return;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    // Compute sum and sum of squares across all particles
    for (int i = 0; i < num_particles; i++) {
        float pos = positions[i * (num_steps + 1) + step_idx];
        sum += pos;
        sum_sq += pos * pos;
    }

    // Compute statistics
    float m = sum / num_particles;
    float m2 = sum_sq / num_particles;

    mean[step_idx] = m;
    msd[step_idx] = m2;
    variance[step_idx] = m2 - m * m;
}

/**
 * Compute mean, MSD, and variance (double precision)
 */
__global__ void compute_stats_f64(
    const double* positions,
    double* mean,
    double* msd,
    double* variance,
    int num_particles,
    int num_steps
) {
    int step_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (step_idx > num_steps) return;

    double sum = 0.0;
    double sum_sq = 0.0;

    for (int i = 0; i < num_particles; i++) {
        double pos = positions[i * (num_steps + 1) + step_idx];
        sum += pos;
        sum_sq += pos * pos;
    }

    double m = sum / num_particles;
    double m2 = sum_sq / num_particles;

    mean[step_idx] = m;
    msd[step_idx] = m2;
    variance[step_idx] = m2 - m * m;
}

/**
 * Compute raw moment of given order
 * 
 * @param positions Input positions [num_particles * (num_steps + 1)]
 * @param moments Output moments [num_steps + 1]
 * @param order Moment order
 * @param num_particles Number of particles
 * @param num_steps Number of time steps
 */
__global__ void compute_raw_moment_f32(
    const float* positions,
    float* moments,
    int order,
    int num_particles,
    int num_steps
) {
    int step_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (step_idx > num_steps) return;

    float sum = 0.0f;

    for (int i = 0; i < num_particles; i++) {
        float pos = positions[i * (num_steps + 1) + step_idx];
        float value = pos;
        for (int j = 1; j < order; j++) {
            value *= pos;
        }
        sum += value;
    }

    moments[step_idx] = sum / num_particles;
}

/**
 * Compute central moment of given order
 * 
 * @param positions Input positions
 * @param means Mean at each time step
 * @param moments Output central moments
 * @param order Moment order
 * @param num_particles Number of particles
 * @param num_steps Number of time steps
 */
__global__ void compute_central_moment_f32(
    const float* positions,
    const float* means,
    float* moments,
    int order,
    int num_particles,
    int num_steps
) {
    int step_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (step_idx > num_steps) return;

    float mean = means[step_idx];
    float sum = 0.0f;

    for (int i = 0; i < num_particles; i++) {
        float pos = positions[i * (num_steps + 1) + step_idx];
        float deviation = pos - mean;
        float value = deviation;
        for (int j = 1; j < order; j++) {
            value *= deviation;
        }
        sum += value;
    }

    moments[step_idx] = sum / num_particles;
}

/**
 * Compute autocorrelation function
 * 
 * @param positions Input positions
 * @param autocorr Output autocorrelation [max_lag + 1]
 * @param max_lag Maximum lag
 * @param num_particles Number of particles
 * @param num_steps Number of time steps
 */
__global__ void compute_autocorrelation_f32(
    const float* positions,
    float* autocorr,
    int max_lag,
    int num_particles,
    int num_steps
) {
    int lag = blockIdx.x * blockDim.x + threadIdx.x;
    if (lag > max_lag || lag > num_steps) return;

    float sum = 0.0f;
    int count = 0;

    // Average over all particles and all valid time pairs
    for (int i = 0; i < num_particles; i++) {
        for (int t = 0; t <= num_steps - lag; t++) {
            float x_t = positions[i * (num_steps + 1) + t];
            float x_t_lag = positions[i * (num_steps + 1) + t + lag];
            sum += x_t * x_t_lag;
            count++;
        }
    }

    autocorr[lag] = (count > 0) ? (sum / count) : 0.0f;
}

/**
 * Compute probability distribution (histogram)
 * 
 * @param positions Input positions (flattened)
 * @param histogram Output histogram counts
 * @param min_val Minimum value for histogram
 * @param max_val Maximum value for histogram
 * @param num_bins Number of histogram bins
 * @param total_samples Total number of samples
 */
__global__ void compute_histogram_f32(
    const float* positions,
    int* histogram,
    float min_val,
    float max_val,
    int num_bins,
    int total_samples
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_samples) return;

    float value = positions[idx];
    if (value < min_val || value > max_val) return;

    float range = max_val - min_val;
    int bin_idx = (int)((value - min_val) / range * num_bins);
    bin_idx = min(bin_idx, num_bins - 1);

    atomicAdd(&histogram[bin_idx], 1);
}

/**
 * Compute ensemble average trajectory
 * 
 * @param positions Input positions [num_particles * (num_steps + 1)]
 * @param ensemble_avg Output ensemble average [num_steps + 1]
 * @param num_particles Number of particles
 * @param num_steps Number of time steps
 */
__global__ void compute_ensemble_average_f32(
    const float* positions,
    float* ensemble_avg,
    int num_particles,
    int num_steps
) {
    int step_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (step_idx > num_steps) return;

    float sum = 0.0f;
    for (int i = 0; i < num_particles; i++) {
        sum += positions[i * (num_steps + 1) + step_idx];
    }

    ensemble_avg[step_idx] = sum / num_particles;
}

/**
 * Compute time-averaged MSD for each particle
 * 
 * @param positions Input positions [num_particles * (num_steps + 1)]
 * @param tamsd Output TAMSD for each particle [num_particles * num_lags]
 * @param num_lags Number of lag times
 * @param num_particles Number of particles
 * @param num_steps Number of time steps
 */
__global__ void compute_tamsd_f32(
    const float* positions,
    float* tamsd,
    int num_lags,
    int num_particles,
    int num_steps
) {
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_idx >= num_particles) return;

    int offset = particle_idx * (num_steps + 1);

    // Compute TAMSD for different lag times
    for (int lag = 1; lag <= num_lags && lag <= num_steps; lag++) {
        float sum = 0.0f;
        int count = 0;

        for (int t = 0; t <= num_steps - lag; t++) {
            float x_t = positions[offset + t];
            float x_t_lag = positions[offset + t + lag];
            float diff = x_t_lag - x_t;
            sum += diff * diff;
            count++;
        }

        tamsd[particle_idx * num_lags + (lag - 1)] = (count > 0) ? (sum / count) : 0.0f;
    }
}

} // extern "C"