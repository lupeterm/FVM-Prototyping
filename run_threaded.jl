include("benchmarking.jl")
case = ARGS[1]
input = ProcessCase(Float64, case)
prep = prepf(input)
pde = Div{Float64, UpwindScheme{Float64}}(UpwindScheme{Float64}(), 1.0) + Laplace(1.0)
batches = getBatches(input)
suite = BenchmarkGroup()
suite["cpu"] = BenchmarkGroup(["cpu"])
suite["cpu"][case] = BenchmarkGroup(["cpu", case])
# suite["cpu"][case]["faceBased"] = BenchmarkGroup(["cpu", "faceBased", case])
suite["cpu"][case]["cellBased"] = BenchmarkGroup(["cpu", "cellBased", case])
suite["cpu"][case]["globalFaceBased"] = BenchmarkGroup(["cpu", "globalFaceBased", case])
suite["cpu"][case]["batchedFace"] = BenchmarkGroup(["cpu", "batchedFace", case])
nthreads = Threads.nthreads()
if nthreads != 1
    # suite["cpu"][case]["faceBased"]["FusedFaceBased-$nthreads"] = @benchmarkable FusedFaceBasedAssembly($input, $prep...,$pde)
    suite["cpu"][case]["globalFaceBased"]["FusedGlobalFaceBased-$nthreads"] = @benchmarkable FusedGlobalFaceBasedAssemblyThreaded($input, $prep...,$pde)
    suite["cpu"][case]["batchedFace"]["FusedBatchedFaceBased-$nthreads"] = @benchmarkable FusedBatchedFaceBasedAssemblyThreaded($input, $prep..., $batches, $pde)
    suite["cpu"][case]["cellBased"]["FusedCellBased-$nthreads"] = @benchmarkable FusedCellBasedAssemblyThreaded($input, $prep...,$pde)
else
    # suite["cpu"][case]["faceBased"]["FusedFaceBased-serial"] = @benchmarkable FusedFaceBasedAssembly($input, $prep...,$pde)
    suite["cpu"][case]["globalFaceBased"]["FusedGlobalFaceBased-serial"] = @benchmarkable FusedGlobalFaceBasedAssemblyThreaded($input, $prep...,$pde)
    suite["cpu"][case]["batchedFace"]["FusedBatchedFaceBased-serial"] = @benchmarkable FusedBatchedFaceBasedAssemblyThreaded($input, $prep..., $batches, $pde)
    suite["cpu"][case]["cellBased"]["FusedCellBased-serial"] = @benchmarkable FusedCellBasedAssemblyThreaded($input, $prep...,$pde)

end
results = run(suite, verbose=true)
processResults(results, "scaling.csv", Float64)