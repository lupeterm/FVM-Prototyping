#!/bin/sh

julia alles.jl
julia -t 2 alles.jl
julia -t 4 alles.jl
julia -t 8 alles.jl
julia -t 16 alles.jl
julia -t 32 alles.jl