using LinearAlgebra
using BenchmarkTools
using StaticArrays
using .Threads
using CUDA



function bench_gc(input::MatrixAssemblyInput, func::Function, case::String, runGC::Bool)
    times = []
    for _ in 1:10
        if runGC
            GC.gc()
        end
        start = time()
        func(input)
        dur = time() - start
        push!(times, dur)
    end
    long = if case == "LDC-S"
        "Lid-Driven Cavity S"
    elseif case == "LDC-M"
        "Lid-Driven Cavity M"
    else
        "WindsorBody"
    end
    ms = mean(times) * 1000
    med = median(times) * 1000
    included = if runGC
        "false"
    else
        "true"
    end
    println("$ms,$case,$long,$(Threads.nthreads()),$(String(Symbol(func))),$included,mean,julia")
    println("$med,$case,$long,$(Threads.nthreads()),$(String(Symbol(func))),$included,median,julia")
end


function bench_phi(input::MatrixAssemblyInput, case::String, runGC::Bool, phiFunc::Function)
    times = []
    for _ in 1:10
        if runGC
            GC.gc()
        end
        start = time()
        DynamicOnlineDivLapFaceBasedAssembly(input, phiFunc)
        dur = time() - start
        push!(times, dur)
    end
    long = if case == "LDC-S"
        "Lid-Driven Cavity S"
    elseif case == "LDC-M"
        "Lid-Driven Cavity M"
    else
        "WindsorBody"
    end
    ms = mean(times) * 1000
    med = median(times) * 1000
    included = if runGC
        "false"
    else
        "true"
    end
    println("time_ms,case_short,case_long,threads,algorithm,incl_gc,metric,language,interpolationMethod")
    println("$ms,$case,$long,$(Threads.nthreads()),BatchedFaceBasedAssembly,$included,mean,julia,$(String(Symbol(phiFunc)))")
    println("$med,$case,$long,$(Threads.nthreads()),BatchedFaceBasedAssembly,$included,median,julia,$(String(Symbol(phiFunc)))")
end

function upwind(ϕf)
    # ϕf Uf ⋅ Sf = 0
    # ϕf is ̇m in the non-versteeg book
    if (ϕf >= 0)
        return 1.0
    end
    return 0.0
end

function centralDifferencing(_)
    return 0.5
end

function precalcWeights(input::MatrixAssemblyInput, div::Function)::Vector{Float32}
    mesh = input.mesh
    U = input.U[2].values
    weights = zeros(Float32, mesh.numInteriorFaces)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor
        U_P = U[iOwner]     
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::Float32 = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = div(ϕf)
        weights[iFace] = weights_f
    end
    return weights
end

