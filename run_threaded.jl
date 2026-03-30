include("benchmarking.jl")
case = ARGS[1]
input = ProcessCase(Float64, case)
prep = prepf(input)
pde = Div{Float64, UpwindScheme{Float64}}(UpwindScheme{Float64}(), 1.0) + Laplace(1.0)
suite = BenchmarkGroup()
suite["cpu"] = BenchmarkGroup(["cpu"])
suite["cpu"][case] = BenchmarkGroup(["cpu", case])
suite["cpu"][case]["faceBased"] = BenchmarkGroup(["cpu", "faceBased", case])
nthreads = Threads.nthreads()
if nthreads != 1
    suite["cpu"][case]["faceBased"]["FusedFaceBased-$nthreads"] = @benchmarkable FusedFaceBasedAssemblyThreaded($input, $prep...,$pde)
else
    suite["cpu"][case]["faceBased"]["FusedFaceBased-serial"] = @benchmarkable FusedFaceBasedAssembly($input, $prep...,$pde)

end
results = run(suite, verbose=true)
processResults(results, "scaling.csv", Float64)