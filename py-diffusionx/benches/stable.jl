using LoopVectorization, Random

function sample_standard_alpha(α, β)
    half_pi = π / 2.0
    tmp = β * tan(α * half_pi)
    v = (rand() - 1 / 2) * 2 * half_pi
    w = randexp()
    b = atan(tmp) / α
    s = (1.0 + tmp * tmp)^(1.0 / (2.0 * α))
    c1 = α * sin(v + b) / cos(v)^(1.0 / α)
    c2 = cos(v - α * (v + b)) / w^(1.0 / α)
    s * c1 * c2
end

function sample_standard_alpha_one(β)
    half_pi = π / 2.0
    v = (rand() - 1 / 2) * 2 * half_pi
    w = randexp()
    c1 = (half_pi + β * v) * tan(v)
    c2 = ((half_pi * w * cos(v)) / log(half_pi + β * v)) * β
    2.0 * (c1 - c2) / π
end

function _sample_stable_alpha(α, β, σ, μ)
    r = sample_standard_alpha(α, β)
    σ * r + μ
end

function _sample_stable_alpha_one(α, β, σ, μ)
    r = sample_standard_alpha_one(β)
    σ * r + μ + 2.0 * β * σ * σ * log(σ) / π
end


function stable_rands(α, β, σ, μ, n)
    res = zeros(n)
    if α == 1.0
        gen = _sample_stable_alpha_one
    else
        gen = _sample_stable_alpha
    end

    # 使用多线程分块计算
    chunk_size = div(n, Threads.nthreads())
    Threads.@threads for tid in 1:Threads.nthreads()
        start_idx = (tid - 1) * chunk_size + 1
        end_idx = tid == Threads.nthreads() ? n : tid * chunk_size

        # 每个线程使用自己的随机数生成器
        rng = Random.default_rng()
        Random.seed!(rng, tid * time_ns())

        # 在每个块内使用 SIMD
        @simd for i in start_idx:end_idx
            @fastmath @inbounds res[i] = gen(α, β, σ, μ)
        end
    end
    res
end
