__inline__ __device__ float warp_reduce_sum(float val)
{
    int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__inline__ __device__ float block_reduce_sum(float val, float *warp_sums)
{
    val = warp_reduce_sum(val);

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0)
    {
        warp_sums[warp_id] = val;
    }

    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0)
    {
        if (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize)
        {
            block_sum = warp_sums[lane];
        }
        block_sum = warp_reduce_sum(block_sum);
    }

    return block_sum;
}