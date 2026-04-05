import Pkg
Pkg.activate(".")
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