import Pkg
Pkg.activate(".")
using LinearAlgebra
using BenchmarkTools
using StaticArrays
using .Threads
using CUDA
include("classes.jl")
include("readMesh.jl")
include("alles.jl")