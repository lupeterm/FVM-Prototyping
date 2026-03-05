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


