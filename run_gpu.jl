include("benchmarking.jl")
include("gpu_abstract.jl")
case = ARGS[1]
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
suite["gpu"] = BenchmarkGroup(["gpu"])
pde = Div{Float64, UpwindScheme{Float64}}(UpwindScheme{Float64}(), 1.0) + Laplace(1.0)
# for case in cases
suite["gpu"][case] = BenchmarkGroup(["gpu", case])
input = ProcessCase(Float64, case)
cellinput = CellInput(input);
facelinkinput = faceLikeInput(input);
faceinput = faceInput(input);
for wg in 64:64:384
    # suite["gpu"][case]["cellBased"]["FusedCellBased-$wg"] = @benchmarkable CellBased($cellinput, $pde, $wg, $input.mesh.numCells)
    suite["gpu"][case]["faceBased"]["FusedFaceBased-$wg"] = @benchmarkable FaceBasedAssembly($facelinkinput[1:end-1]..., $pde, $wg, $input.mesh.numInteriorFaces)
    suite["gpu"][case]["batchedFace"]["FusedBatchedFace-$wg"] = @benchmarkable FusedBatchedAssembly($facelinkinput..., $pde, $wg, $input.mesh.numInteriorFaces)
    suite["gpu"][case]["globalFaceBased"]["FusedGlobalFaceBased-$wg"] = @benchmarkable GlobalFaceAssembly($faceinput..., $pde, $wg, $length(input.mesh.faces))
end
# end
results = run(suite, verbose=true)
processResults(results, "gpuScaling.csv", Float64)