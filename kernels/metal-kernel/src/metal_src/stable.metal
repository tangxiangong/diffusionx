#include <metal_stdlib>
using namespace metal;

// Philox RNG 状态
struct PhiloxState {
    uint4 ctr;
    uint2 key;
};

// Philox 4x32 轮函数
inline uint4 philox_round(uint4 ctr, uint2 key) {
    const uint M0 = 0xD2511F53;
    const uint M1 = 0xCD9E8D57;
    
    uint hi0 = mulhi(M0, ctr.x);
    uint lo0 = M0 * ctr.x;
    uint hi1 = mulhi(M1, ctr.z);
    uint lo1 = M1 * ctr.z;
    
    return uint4(hi1 ^ ctr.y ^ key.x, lo1,
                 hi0 ^ ctr.w ^ key.y, lo0);
}

// Philox 生成随机数
inline uint4 philox_generate(thread PhiloxState& state) {
    const uint W0 = 0x9E3779B9;
    const uint W1 = 0xBB67AE85;
    
    uint4 ctr = state.ctr;
    uint2 key = state.key;
    
    // 10 轮
    for (int i = 0; i < 10; i++) {
        ctr = philox_round(ctr, key);
        key.x += W0;
        key.y += W1;
    }
    
    state.ctr.x += 1;
    if (state.ctr.x == 0) {
        state.ctr.y += 1;
        if (state.ctr.y == 0) {
            state.ctr.z += 1;
            if (state.ctr.z == 0) {
                state.ctr.w += 1;
            }
        }
    }
    
    return ctr;
}

// 生成 [0, 1) 均匀分布
inline float uniform_float(thread PhiloxState& state) {
    uint4 rand = philox_generate(state);
    return float(rand.x) / 4294967296.0f;
}

// Stable 分布采样（Chambers-Mallows-Stuck 方法）
kernel void generate_stable(
    device float* output [[buffer(0)]],
    constant float& alpha [[buffer(1)]],
    constant float& beta [[buffer(2)]],
    constant uint& seed [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    // 初始化 RNG
    PhiloxState state;
    state.ctr = uint4(gid, 0, 0, 0);
    state.key = uint2(seed, seed >> 32);
    
    // 生成均匀分布 U ~ Uniform(-π/2, π/2)
    float u = uniform_float(state) * M_PI_F - M_PI_2_F;
    
    // 生成指数分布 W ~ Exp(1)
    float w = -log(uniform_float(state));
    
    float result;
    
    // alpha = 1 的特殊情况
    if (abs(alpha - 1.0f) < 1e-6f) {
        float xi = M_PI_2_F + beta * u;
        result = (2.0f / M_PI_F) * (xi * tan(u) - beta * log(w));
    } else {
        // 一般情况
        float zeta = -beta * tan(M_PI_F * alpha / 2.0f);
        float xi = atan(zeta) / alpha;
        
        float part1 = sin(alpha * (u + xi));
        float part2 = pow(cos(u), 1.0f / alpha);
        float part3 = cos((1.0f - alpha) * (u + xi)) / w;
        float part3_pow = pow(part3, (1.0f - alpha) / alpha);
        
        result = part1 / part2 * part3_pow;
    }
    
    output[gid] = result;
}
