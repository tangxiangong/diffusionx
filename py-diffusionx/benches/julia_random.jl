using Random, BenchmarkTools

N = 100_000_000;
println("标准正态分布")
@benchmark randn($N);
println("[0, 1] 均匀分布")
@benchmark rand($N);
