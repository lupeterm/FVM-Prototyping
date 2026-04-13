include("benchmarking.jl")
include("soa_cpu.jl")
include("cpu_threaded.jl")
# case = ARGS[1]
cases = [
    # "cases/Lid-Driven-Cavities/10/",
    # "cases/Lid-Driven-Cavities/20/",
    # "cases/Lid-Driven-Cavities/30/",
    # "cases/Lid-Driven-Cavities/40/",
    # "cases/Lid-Driven-Cavities/50/",
    # "cases/Lid-Driven-Cavities/60/",
    # "cases/Lid-Driven-Cavities/70/",
    # "cases/Lid-Driven-Cavities/80/",
    # "cases/Lid-Driven-Cavities/90/",
    # "cases/Lid-Driven-Cavities/100/",
    # "cases/Lid-Driven-Cavities/120/",
    # "cases/Lid-Driven-Cavities/140/",
    # "cases/Lid-Driven-Cavities/160/",
    # "cases/Lid-Driven-Cavities/180/",
    "cases/Lid-Driven-Cavities/200/",
]
suite = BenchmarkGroup()
suite["cpu"] = BenchmarkGroup(["cpu"])
pde = Div{Float64, UpwindScheme{Float64}}(UpwindScheme{Float64}(), 1.0) + Laplace(1.0)
for case in cases
    suite["cpu"][case] = BenchmarkGroup(["cpu", case])
    input = ProcessCase(Float64, case)
    soaInput = toSOAInput(input)
    prep = prepf(input)
    batches = getBatches(input)
    nthreads = Threads.nthreads()
    if nthreads > 1
        suite["cpu"][case]["faceBased"]["SOAFusedFaceBasedAssemblyThreaded-$nthreads"] = @benchmarkable SOAFusedFaceBasedAssemblyThreaded($soaInput, $prep..., $pde)
        suite["cpu"][case]["globalFaceBased"]["FusedGlobalFaceBased-$nthreads"] = @benchmarkable SOAFusedGlobalFaceBasedAssemblyThreaded($soaInput, $prep...,$pde)
        suite["cpu"][case]["batchedFace"]["FusedBatchedFaceBased-$nthreads"] = @benchmarkable SOAFusedBatchedFaceBasedAssemblyThreaded($soaInput, $prep..., $batches, $pde)
        suite["cpu"][case]["cellBased"]["FusedCellBased-$nthreads"] = @benchmarkable SOAFusedCellBasedAssemblyThreaded($soaInput, $prep...,$pde)
    else
        # suite["cpu"][case]["faceBased"]["SOAFusedFaceBasedAssembly-serial"] = @benchmarkable SOAFusedFaceBasedAssembly($soaInput, $prep..., $pde)
        # suite["cpu"][case]["globalFaceBased"]["FusedGlobalFaceBased-serial"] = @benchmarkable SOAFusedGlobalFaceBasedAssembly($soaInput, $prep...,$pde)
        # suite["cpu"][case]["batchedFace"]["FusedBatchedFaceBased-serial"] = @benchmarkable SOAFusedBatchedFaceBasedAssembly($soaInput, $prep..., $batches, $pde)
        suite["cpu"][case]["cellBased"]["FusedCellBased-serial"] = @benchmarkable SOAFusedCellBasedAssembly($soaInput, $prep...,$pde)
    end
    GC.gc()
end
results = run(suite, verbose=true)
processResults(results, "cpuScaling.csv", Float64)