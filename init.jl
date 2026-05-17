import Pkg
Pkg.activate(@__DIR__)
using LinearAlgebra
using BenchmarkTools
using StaticArrays
using .Threads
using CUDA
using Atomix
# using Metal
include("classes.jl")
include("readMesh.jl")
include("alles.jl")
include("operators.jl")
function use_gpu_ptr(p::Ptr{Cvoid}, s::Int64)
    v = unsafe_wrap(CuArray, reinterpret(CuPtr{Float64}, p), s)
    v[1:3] = [1.0,2.0, 3.0]
end
