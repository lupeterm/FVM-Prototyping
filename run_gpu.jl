include("benchmarking.jl")
case = ARGS[1]
input = ProcessCase(Float64, case)
bench_gpu(input, Float64, case)