using Random, BenchmarkTools

include("stable.jl")
stable_rands(0.7, 1.0, 1.0, 0.0, 10)

N = 10_000_000;
println("标准正态分布")
@btime randn($N);
println("[0, 1] 均匀分布")
@btime rand($N);
println("稳定分布")
@btime stable_rands(0.7, 1.0, 1.0, 0.0, $N);
