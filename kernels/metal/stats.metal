#include <metal_stdlib>
using namespace metal;

/**
 * Metal Shader for Monte Carlo Statistics Computation
 *
 * This shader computes statistical quantities (mean, MSD, variance)
 * directly on GPU from simulated particle trajectories.
 */

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
 * @param step_idx Thread index (time step)
 */
kernel void compute_stats(
    device const float* positions [[buffer(0)]],
    device float* mean [[buffer(1)]],
    device float* msd [[buffer(2)]],
    device float* variance [[buffer(3)]],
    constant int& num_particles [[buffer(4)]],
    constant int& num_steps [[buffer(5)]],
    uint step_idx [[thread_position_in_grid]]
) {
    if (step_idx > uint(num_steps)) return;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    // Compute sum and sum of squares across all particles
    for (int i = 0; i < num_particles; i++) {
        float pos = positions[i * (num_steps + 1) + step_idx];
        sum += pos;
        sum_sq += pos * pos;
    }

    // Compute statistics
    float m = sum / float(num_particles);
    float m2 = sum_sq / float(num_particles);

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
kernel void compute_raw_moment(
    device const float* positions [[buffer(0)]],
    device float* moments [[buffer(1)]],
    constant int& order [[buffer(2)]],
    constant int& num_particles [[buffer(3)]],
    constant int& num_steps [[buffer(4)]],
    uint step_idx [[thread_position_in_grid]]
) {
    if (step_idx > uint(num_steps)) return;

    float sum = 0.0f;

    for (int i = 0; i < num_particles; i++) {
        float pos = positions[i * (num_steps + 1) + step_idx];
        float value = pos;
        for (int j = 1; j < order; j++) {
            value *= pos;
        }
        sum += value;
    }

    moments[step_idx] = sum / float(num_particles);
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
kernel void compute_central_moment(
    device const float* positions [[buffer(0)]],
    device const float* means [[buffer(1)]],
    device float* moments [[buffer(2)]],
    constant int& order [[buffer(3)]],
    constant int& num_particles [[buffer(4)]],
    constant int& num_steps [[buffer(5)]],
    uint step_idx [[thread_position_in_grid]]
) {
    if (step_idx > uint(num_steps)) return;

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

    moments[step_idx] = sum / float(num_particles);
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
kernel void compute_autocorrelation(
    device const float* positions [[buffer(0)]],
    device float* autocorr [[buffer(1)]],
    constant int& max_lag [[buffer(2)]],
    constant int& num_particles [[buffer(3)]],
    constant int& num_steps [[buffer(4)]],
    uint lag [[thread_position_in_grid]]
) {
    if (lag > uint(max_lag) || lag > uint(num_steps)) return;

    float sum = 0.0f;
    int count = 0;

    // Average over all particles and all valid time pairs
    for (int i = 0; i < num_particles; i++) {
        for (int t = 0; t <= num_steps - int(lag); t++) {
            float x_t = positions[i * (num_steps + 1) + t];
            float x_t_lag = positions[i * (num_steps + 1) + t + lag];
            sum += x_t * x_t_lag;
            count++;
        }
    }

    autocorr[lag] = (count > 0) ? (sum / float(count)) : 0.0f;
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
kernel void compute_histogram(
    device const float* positions [[buffer(0)]],
    device atomic_int* histogram [[buffer(1)]],
    constant float& min_val [[buffer(2)]],
    constant float& max_val [[buffer(3)]],
    constant int& num_bins [[buffer(4)]],
    constant int& total_samples [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= uint(total_samples)) return;

    float value = positions[idx];
    if (value < min_val || value > max_val) return;

    float range = max_val - min_val;
    int bin_idx = int((value - min_val) / range * float(num_bins));
    bin_idx = min(bin_idx, num_bins - 1);

    atomic_fetch_add_explicit(&histogram[bin_idx], 1, memory_order_relaxed);
}

/**
 * Compute ensemble average trajectory
 *
 * @param positions Input positions [num_particles * (num_steps + 1)]
 * @param ensemble_avg Output ensemble average [num_steps + 1]
 * @param num_particles Number of particles
 * @param num_steps Number of time steps
 */
kernel void compute_ensemble_average(
    device const float* positions [[buffer(0)]],
    device float* ensemble_avg [[buffer(1)]],
    constant int& num_particles [[buffer(2)]],
    constant int& num_steps [[buffer(3)]],
    uint step_idx [[thread_position_in_grid]]
) {
    if (step_idx > uint(num_steps)) return;

    float sum = 0.0f;
    for (int i = 0; i < num_particles; i++) {
        sum += positions[i * (num_steps + 1) + step_idx];
    }

    ensemble_avg[step_idx] = sum / float(num_particles);
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
kernel void compute_tamsd(
    device const float* positions [[buffer(0)]],
    device float* tamsd [[buffer(1)]],
    constant int& num_lags [[buffer(2)]],
    constant int& num_particles [[buffer(3)]],
    constant int& num_steps [[buffer(4)]],
    uint particle_idx [[thread_position_in_grid]]
) {
    if (particle_idx >= uint(num_particles)) return;

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

        tamsd[particle_idx * num_lags + (lag - 1)] = (count > 0) ? (sum / float(count)) : 0.0f;
    }
}

/**
 * Reduce results from multiple work groups (sum reduction)
 * Used for computing global statistics
 */
kernel void reduce_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_data[256];

    // Load data into shared memory
    if (gid < n) {
        shared_data[lid] = input[gid];
    } else {
        shared_data[lid] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_data[lid] += shared_data[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (lid == 0) {
        output[gid / group_size] = shared_data[0];
    }
}
