include("benchmarking.jl")
include("soa_cpu.jl")
include("cpu_threaded.jl")
case = ARGS[1]
println("Running case $case")
suite = BenchmarkGroup()
suite["cpu"] = BenchmarkGroup(["cpu"])
pde = Div{Float64, upwind{Float64}}(upwind{Float64}(), 1.0) + Laplace(1.0)

suite["cpu"][case] = BenchmarkGroup(["cpu", case])
input = ProcessCase(Float64, "$case/case")
numCells = input.mesh.numCells
soaInput = toSOAInput(input)
prep = prepf(input)
casename = "LDC-$numCells"
batches = getBatches(input)
nthreads = Threads.nthreads()
if nthreads > 1
    suite["cpu"][casename]["faceBased"]["SOAFusedFaceBasedAssemblyThreaded-$nthreads"] = @benchmarkable SOAFusedFaceBasedAssemblyThreaded($soaInput, $prep..., $pde)
    suite["cpu"][casename]["globalFaceBased"]["FusedGlobalFaceBased-$nthreads"] = @benchmarkable SOAFusedGlobalFaceBasedAssemblyThreaded($soaInput, $prep...,$pde)
    suite["cpu"][casename]["batchedFace"]["FusedBatchedFaceBased-$nthreads"] = @benchmarkable SOAFusedBatchedFaceBasedAssemblyThreaded($soaInput, $prep..., $batches, $pde)
    suite["cpu"][casename]["cellBased"]["FusedCellBased-$nthreads"] = @benchmarkable SOAFusedCellBasedAssemblyThreaded($soaInput, $prep...,$pde)
else
    suite["cpu"][casename]["faceBased"]["SOAFusedFaceBasedAssembly-serial"] = @benchmarkable SOAFusedFaceBasedAssembly($soaInput, $prep..., $pde)
    suite["cpu"][casename]["globalFaceBased"]["FusedGlobalFaceBased-serial"] = @benchmarkable SOAFusedGlobalFaceBasedAssembly($soaInput, $prep...,$pde)
    suite["cpu"][casename]["batchedFace"]["FusedBatchedFaceBased-serial"] = @benchmarkable SOAFusedBatchedFaceBasedAssembly($soaInput, $prep..., $batches, $pde)
    suite["cpu"][casename]["cellBased"]["FusedCellBased-serial"] = @benchmarkable SOAFusedCellBasedAssembly($soaInput, $prep...,$pde)
end

results = run(suite, verbose=true)
processResults(results, "CPU_huge.csv", Float64)
