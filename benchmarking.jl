include("init.jl")
include("faceBased.jl")
include("globalFaceBased.jl")
include("cellBased.jl")
include("gpu_faceBased.jl")
include("gpu_batchedFace.jl")
include("gpu_cellBased.jl")
using BenchmarkTools
const CASES = [
    ("cases/LDC-S/", "LDC-S", "Lid-Driven-Cavity S")
    # ("cases/Wind", "Wind", "WindsorBody")
    # ("cases/LDC-M", "LDC-M", "Lid-Driven-Cavity M")
]

struct Result
    time_mean_ms::Float32
    time_median_ms::Float32
    gc_time_mean_ms::Float32
    gc_time_median_ms::Float32
    case_short::String
    case_long::String
    strategy::String
    variant::String
    language::String
end

ResultToCsvRow(r::Result,cpu::Bool) = "$(r.time_mean_ms),$(r.time_median_ms),$(r.gc_time_mean_ms),$(r.gc_time_median_ms),$(r.case_short),$(r.case_long),$(r.strategy),$(r.variant),$(r.language),$cpu,$(!cpu)\n"

function bench()
    open("results.csv", "a") do io
        write(io, join("time_mean,time_median,gc_time_mean,gc_time_median,case_short,case_long,strategy,variant,language\n"))
        write(io, join(results, "\n"))

    end
    for (case_path, case_short, case_long) in CASES
        input = ProcessCase(case_path)
        for (func, funcName) in REGISTERED_FUNCS
            println("Benchmarking $funcName on $case_long")
            GC.gc()
            b = @benchmark $func($input)
            r = Result(
                mean(b).time / 1e6,
                median(b).time / 1e6,
                mean(b).gctime / 1e6,
                median(b).gctime / 1e6,
                case_short,
                case_long,
                funcName,
                "",
                "julia"
            )
            open("results.csv", "a") do io
                write(io, join(results, "\n"))
            end
        end
    end
end
struct GpuResult
    time_mean::Float32
    time_median::Float32
    case_short::String
    case_long::String
    algorithm::String
    language::String
    threads::Int32
    iBlocks::Int32
    bBlocks::Int32
end
GpuResultToCsvRow(r::GpuResult) = "$(r.time_mean),$(r.time_median),$(r.case_short),$(r.case_long),$(r.algorithm),$(r.language),$(r.threads),$(r.iBlocks),$(r.bBlocks)"

function bench_gpu()
    results = []
    gpu_funcs = [(gpu_prepareFaceBased, faceBasedRunner), (gpu_prepareBatchedFaceBased, batchedFaceBasedRunner)]
    for (case_path, case_short, case_long) in CASES
        input = ProcessCase(case_path)
        for (prepper, runner) in gpu_funcs
            prep = prepper(input)
            println("Benchmarking $(String(Symbol(runner))) on $case_long")
            b = @benchmark $runner($prep...)
            threads::Int32 = 256
            N = input.mesh.numInteriorFaces
            iBlocks::Int32 = cld(N, threads)
            M = input.mesh.numBoundaryFaces
            bBlocks = cld(M, threads)

            r = GpuResult(
                mean(b).time / 1e6,
                median(b).time / 1e6,
                case_short,
                case_long,
                String(Symbol(runner)),
                "julia",
                threads,
                iBlocks,
                bBlocks
            )
            push!(results, GpuResultToCsvRow(r))
        end
    end
end


function bench_all()
    suite = BenchmarkGroup()
    suite["cpu"] = BenchmarkGroup(["cpu"])
    suite["gpu"] = BenchmarkGroup(["gpu"])
    for (case_path, case_short, case_long) in CASES
        println("loading $case_short")
        input = ProcessCase(case_path)

        suite["cpu"][case_short] = BenchmarkGroup(["cpu", case_short])
        suite["cpu"][case_short]["faceBased"] = BenchmarkGroup(["cpu", "faceBased", case_long])
        suite["cpu"][case_short]["faceBased"]["LaplaceOnly"] = @benchmarkable LaplaceOnlyFaceBasedAssembly($input)
        suite["cpu"][case_short]["faceBased"]["DivOnlyPrecalculatedWeightsUpwind"] = @benchmarkable DivOnlyPrecalculatedWeightsUpwindFaceBasedAssembly($input)
        suite["cpu"][case_short]["faceBased"]["DivOnlyPrecalculatedWeightsCDF"] = @benchmarkable DivOnlyPrecalculatedWeightsCDFFaceBasedAssembly($input)
        suite["cpu"][case_short]["faceBased"]["DivOnlyHardCodedUpwind"] = @benchmarkable DivOnlyHardcodedUpwindFaceBasedAssembly($input)
        suite["cpu"][case_short]["faceBased"]["DivOnlyHardCodedCDF"] = @benchmarkable DivOnlyHardcodedCDFFaceBasedAssembly($input)
        suite["cpu"][case_short]["faceBased"]["DivOnlyDynamicCDF"] = @benchmarkable DivOnlyDynamicCDFFaceBasedAssembly($input, centralDifferencing)
        suite["cpu"][case_short]["faceBased"]["DivOnlyDynamicUpwind"] = @benchmarkable DivOnlyDynamicUpwindFaceBasedAssembly($input, upwind)
        suite["cpu"][case_short]["faceBased"]["PrecalculatedWeightsUpwind"] = @benchmarkable PrecalculatedWeightsUpwindFaceBasedAssembly($input)
        suite["cpu"][case_short]["faceBased"]["PrecalculatedWeightsCDF"] = @benchmarkable PrecalculatedWeightsCDFFaceBasedAssembly($input)
        suite["cpu"][case_short]["faceBased"]["HardCodedUpwind"] = @benchmarkable HardcodedUpwindFaceBasedAssembly($input)
        suite["cpu"][case_short]["faceBased"]["HardCodedCDF"] = @benchmarkable HardcodedCDFFaceBasedAssembly($input)
        suite["cpu"][case_short]["faceBased"]["DynamicCDF"] = @benchmarkable DynamicCDFFaceBasedAssembly($input, centralDifferencing)
        suite["cpu"][case_short]["faceBased"]["DynamicUpwind"] = @benchmarkable DynamicUpwindFaceBasedAssembly($input, upwind)

        suite["cpu"][case_short]["cellBased"] = BenchmarkGroup(["cpu", "cellBased", case_long])
        suite["cpu"][case_short]["cellBased"]["LaplaceOnly"] = @benchmarkable LaplaceOnlyCellBasedAssembly($input)
        suite["cpu"][case_short]["cellBased"]["PrecalculatedWeightsUpwind"] = @benchmarkable PrecalculatedWeightsUpwindCellBasedAssembly($input)
        suite["cpu"][case_short]["cellBased"]["PrecalculatedWeightsCDF"] = @benchmarkable PrecalculatedWeightsCDFCellBasedAssembly($input)
        suite["cpu"][case_short]["cellBased"]["DynamicCDF"] = @benchmarkable DynamicCDFCellBasedAssembly($input, centralDifferencing)
        suite["cpu"][case_short]["cellBased"]["DynamicUpwind"] = @benchmarkable DynamicUpwindCellBasedAssembly($input, upwind)
        suite["cpu"][case_short]["cellBased"]["DivOnlyPrecalculatedWeightsUpwind"] = @benchmarkable DivOnlyPrecalculatedWeightsUpwindCellBasedAssembly($input)
        suite["cpu"][case_short]["cellBased"]["DivOnlyPrecalculatedWeightsCDF"] = @benchmarkable DivOnlyPrecalculatedWeightsCDFCellBasedAssembly($input)
        suite["cpu"][case_short]["cellBased"]["DivOnlyHardCodedUpwind"] = @benchmarkable DivOnlyHardcodedUpwindCellBasedAssembly($input)
        suite["cpu"][case_short]["cellBased"]["DivOnlyHardCodedCDF"] = @benchmarkable DivOnlyHardcodedCDFCellBasedAssembly($input)
        suite["cpu"][case_short]["cellBased"]["DivOnlyDynamicCDF"] = @benchmarkable DivOnlyDynamicCDFCellBasedAssembly($input, centralDifferencing)
        suite["cpu"][case_short]["cellBased"]["DivOnlyDynamicUpwind"] = @benchmarkable DivOnlyDynamicUpwindCellBasedAssembly($input, upwind)
        suite["cpu"][case_short]["cellBased"]["HardCodedUpwind"] = @benchmarkable HardcodedUpwindCellBasedAssembly($input)
        suite["cpu"][case_short]["cellBased"]["HardCodedCDF"] = @benchmarkable HardcodedCDFCellBasedAssembly($input)

        suite["cpu"][case_short]["globalFaceBased"] = BenchmarkGroup(["cpu", "globalFaceBased", case_long])
        suite["cpu"][case_short]["globalFaceBased"]["LaplaceOnly"] = @benchmarkable LaplaceOnlyGlobalFaceBasedAssembly($input)
        suite["cpu"][case_short]["globalFaceBased"]["DivOnlyPrecalculatedWeightsUpwind"] = @benchmarkable DivOnlyPrecalculatedWeightsUpwindGlobalFaceBasedAssembly($input)
        suite["cpu"][case_short]["globalFaceBased"]["DivOnlyPrecalculatedWeightsCDF"] = @benchmarkable DivOnlyPrecalculatedWeightsCDFGlobalFaceBasedAssembly($input)
        suite["cpu"][case_short]["globalFaceBased"]["DivOnlyHardCodedUpwind"] = @benchmarkable DivOnlyHardcodedUpwindGlobalFaceBasedAssembly($input)
        suite["cpu"][case_short]["globalFaceBased"]["DivOnlyHardCodedCDF"] = @benchmarkable DivOnlyHardcodedCDFGlobalFaceBasedAssembly($input)
        suite["cpu"][case_short]["globalFaceBased"]["DivOnlyDynamicCDF"] = @benchmarkable DivOnlyDynamicCDFGlobalFaceBasedAssembly($input, centralDifferencing)
        suite["cpu"][case_short]["globalFaceBased"]["DivOnlyDynamicUpwind"] = @benchmarkable DivOnlyDynamicUpwindGlobalFaceBasedAssembly($input, upwind)
        suite["cpu"][case_short]["globalFaceBased"]["PrecalculatedWeightsUpwind"] = @benchmarkable PrecalculatedWeightsUpwindGlobalFaceBasedAssembly($input)
        suite["cpu"][case_short]["globalFaceBased"]["PrecalculatedWeightsCDF"] = @benchmarkable PrecalculatedWeightsCDFGlobalFaceBasedAssembly($input)
        suite["cpu"][case_short]["globalFaceBased"]["HardCodedUpwind"] = @benchmarkable HardcodedUpwindGlobalFaceBasedAssembly($input)
        suite["cpu"][case_short]["globalFaceBased"]["HardCodedCDF"] = @benchmarkable HardcodedCDFGlobalFaceBasedAssembly($input)
        suite["cpu"][case_short]["globalFaceBased"]["DynamicCDF"] = @benchmarkable DynamicCDFGlobalFaceBasedAssembly($input, centralDifferencing)
        suite["cpu"][case_short]["globalFaceBased"]["DynamicUpwind"] = @benchmarkable DynamicUpwindGlobalFaceBasedAssembly($input, upwind)

        # suite["gpu"][case_short] = BenchmarkGroup(["gpu", case_short])
        # suite["gpu"][case_short]["batchedFace"] = BenchmarkGroup(["gpu", "batchedFace", case_long])
        # suite["gpu"][case_short]["batchedFace"]["LaplaceOnly"] = @benchmarkable gpu_LaplaceOnlyBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyPrecalculatedWeightsUpwind"] = @benchmarkable gpu_DivOnlyPrecalculatedWeightsUpwindBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyPrecalculatedWeightsCDF"] = @benchmarkable gpu_DivOnlyPrecalculatedWeightsCDFBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyHardCodedUpwind"] = @benchmarkable gpu_DivOnlyHardcodedUpwindBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyHardCodedCDF"] = @benchmarkable gpu_DivOnlyHardcodedCDFBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyDynamicCDF"] = @benchmarkable gpu_DivOnlyDynamicCDFBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyDynamicUpwind"] = @benchmarkable gpu_DivOnlyDynamicUpwindBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["PrecalculatedWeightsUpwind"] = @benchmarkable gpu_PrecalculatedWeightsUpwindBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["PrecalculatedWeightsCDF"] = @benchmarkable gpu_PrecalculatedWeightsCDFBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["HardCodedUpwind"] = @benchmarkable gpu_HardcodedUpwindBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["HardCodedCDF"] = @benchmarkable gpu_HardcodedCDFBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["DynamicCDF"] = @benchmarkable gpu_DynamicCDFBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["DynamicUpwind"] = @benchmarkable gpu_DynamicUpwindBatchedFaceAssembly($input)
        
        # suite["gpu"][case_short]["faceBased"] = BenchmarkGroup(["gpu", "faceBased", case_long])
        # suite["gpu"][case_short]["faceBased"]["LaplaceOnly"] = @benchmarkable gpu_LaplaceOnlyFaceBasedAssembly($input)
        # suite["gpu"][case_short]["faceBased"]["DivOnlyPrecalculatedWeightsUpwind"] = @benchmarkable gpu_DivOnlyPrecalculatedWeightsFaceBasedAssemblyRunner($prep..., CuArray(input.weightsUpwind))
        # suite["gpu"][case_short]["faceBased"]["DivOnlyPrecalculatedWeightsCDF"] = @benchmarkable gpu_DivOnlyPrecalculatedWeightsCDFFaceBasedAssembly($prep..., CuArray(input.weightsUpwind))
        # suite["gpu"][case_short]["faceBased"]["DivOnlyHardCodedUpwind"] = @benchmarkable gpu_DivOnlyHardcodedUpwindFaceBasedAssembly($input)
        # suite["gpu"][case_short]["faceBased"]["DivOnlyHardCodedCDF"] = @benchmarkable gpu_DivOnlyHardcodedCDFFaceBasedAssembly($input)
        # suite["gpu"][case_short]["faceBased"]["DivOnlyDynamicCDF"] = @benchmarkable gpu_DivOnlyDynamicCDFFaceBasedAssembly($input)
        # suite["gpu"][case_short]["faceBased"]["DivOnlyDynamicUpwind"] = @benchmarkable gpu_DivOnlyDynamicUpwindFaceBasedAssembly($input)
        # suite["gpu"][case_short]["faceBased"]["PrecalculatedWeightsUpwind"] = @benchmarkable gpu_PrecalculatedWeightsUpwindFaceBasedAssembly($input)
        # suite["gpu"][case_short]["faceBased"]["PrecalculatedWeightsCDF"] = @benchmarkable gpu_PrecalculatedWeightsCDFFaceBasedAssembly($input)
        # suite["gpu"][case_short]["faceBased"]["HardCodedUpwind"] = @benchmarkable gpu_HardcodedUpwindFaceBasedAssembly($input)
        # suite["gpu"][case_short]["faceBased"]["HardCodedCDF"] = @benchmarkable gpu_HardcodedCDFFaceBasedAssembly($input)
        # suite["gpu"][case_short]["faceBased"]["DynamicCDF"] = @benchmarkable gpu_DynamicCDFFaceBasedAssembly($input)
        # suite["gpu"][case_short]["faceBased"]["DynamicUpwind"] = @benchmarkable gpu_DynamicUpwindFaceBasedAssembly($input)
        
        # suite["gpu"][case_short]["cellBased"] = BenchmarkGroup(["gpu", "cellBased", case_long])
        # suite["gpu"][case_short]["cellBased"]["LaplaceOnly"] = @benchmarkable LaplaceOnlyCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["DivOnlyPrecalculatedWeightsUpwind"] = @benchmarkable DivOnlyPrecalculatedWeightsUpwindCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["DivOnlyPrecalculatedWeightsCDF"] = @benchmarkable DivOnlyPrecalculatedWeightsCDFCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["DivOnlyHardCodedUpwind"] = @benchmarkable DivOnlyHardcodedUpwindCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["DivOnlyHardCodedCDF"] = @benchmarkable DivOnlyHardcodedCDFCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["DivOnlyDynamicCDF"] = @benchmarkable DivOnlyDynamicCDFCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["DivOnlyDynamicUpwind"] = @benchmarkable DivOnlyDynamicUpwindCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["PrecalculatedWeightsUpwind"] = @benchmarkable PrecalculatedWeightsUpwindCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["PrecalculatedWeightsCDF"] = @benchmarkable PrecalculatedWeightsCDFCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["HardCodedUpwind"] = @benchmarkable HardcodedUpwindCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["HardCodedCDF"] = @benchmarkable HardcodedCDFCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["DynamicCDF"] = @benchmarkable DynamicCDFCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["DynamicUpwind"] = @benchmarkable DynamicUpwindCellBasedAssembly($input)
        results = run(suite, verbose = true)
        processResults(results)
    end
    # return suite
end


function bench_gpu(i = nothing)
    suite = BenchmarkGroup()
    suite["gpu"] = BenchmarkGroup(["gpu"])
    for (case_path, case_short, case_long) in CASES
        println("loading $case_short")
        input = if !isnothing(i) i else ProcessCase(case_path) end
        # input = ProcessCase(case_path)

        suite["gpu"][case_short] = BenchmarkGroup(["gpu", case_short])
        # suite["gpu"][case_short]["batchedFace"] = BenchmarkGroup(["gpu", "batchedFace", case_long])
        # suite["gpu"][case_short]["batchedFace"]["LaplaceOnly"] = @benchmarkable gpu_LaplaceOnlyBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyPrecalculatedWeightsUpwind"] = @benchmarkable gpu_DivOnlyPrecalculatedWeightsUpwindBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyPrecalculatedWeightsCDF"] = @benchmarkable gpu_DivOnlyPrecalculatedWeightsCDFBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyHardCodedUpwind"] = @benchmarkable gpu_DivOnlyHardcodedUpwindBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyHardCodedCDF"] = @benchmarkable gpu_DivOnlyHardcodedCDFBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyDynamicCDF"] = @benchmarkable gpu_DivOnlyDynamicCDFBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyDynamicUpwind"] = @benchmarkable gpu_DivOnlyDynamicUpwindBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["PrecalculatedWeightsUpwind"] = @benchmarkable gpu_PrecalculatedWeightsUpwindBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["PrecalculatedWeightsCDF"] = @benchmarkable gpu_PrecalculatedWeightsCDFBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["HardCodedUpwind"] = @benchmarkable gpu_HardcodedUpwindBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["HardCodedCDF"] = @benchmarkable gpu_HardcodedCDFBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["DynamicCDF"] = @benchmarkable gpu_DynamicCDFBatchedFaceAssembly($input)
        # suite["gpu"][case_short]["batchedFace"]["DynamicUpwind"] = @benchmarkable gpu_DynamicUpwindBatchedFaceAssembly($input)
        prep = gpu_prepareFaceBased(input)
        suite["gpu"][case_short]["faceBased"] = BenchmarkGroup(["gpu", "faceBased", case_long])
        suite["gpu"][case_short]["faceBased"]["LaplaceOnly"] = @benchmarkable gpu_LaplaceOnlyFaceBasedAssemblyRunner($prep...)
        suite["gpu"][case_short]["faceBased"]["DivOnlyPrecalculatedWeightsUpwind"] = @benchmarkable gpu_DivOnlyPrecalculatedWeightsFaceBasedAssemblyRunner($prep..., CuArray(input.weightsUpwind))
        suite["gpu"][case_short]["faceBased"]["DivOnlyPrecalculatedWeightsCDF"] = @benchmarkable gpu_DivOnlyPrecalculatedWeightsFaceBasedAssemblyRunner($prep..., CuArray(input.weightsUpwind))
        suite["gpu"][case_short]["faceBased"]["DivOnlyHardCodedUpwind"] = @benchmarkable gpu_DivOnlyHardcodedDivFaceBasedAssemblyRunner($prep..., "Upwind")
        suite["gpu"][case_short]["faceBased"]["DivOnlyHardCodedCDF"] = @benchmarkable gpu_DivOnlyHardcodedDivFaceBasedAssemblyRunner($prep..., "CDF")
        suite["gpu"][case_short]["faceBased"]["DivOnlyDynamicCDF"] = @benchmarkable gpu_DivOnlyDynamicFaceBasedAssemblyRunner($prep..., centralDifferencing)
        suite["gpu"][case_short]["faceBased"]["DivOnlyDynamicUpwind"] = @benchmarkable gpu_DivOnlyDynamicFaceBasedAssemblyRunner($prep..., upwind)
        suite["gpu"][case_short]["faceBased"]["PrecalculatedWeightsUpwind"] = @benchmarkable gpu_PrecalculatedWeightsFaceBasedAssemblyRunner($prep..., CuArray(input.weightsUpwind))
        suite["gpu"][case_short]["faceBased"]["PrecalculatedWeightsCDF"] = @benchmarkable gpu_PrecalculatedWeightsFaceBasedAssemblyRunner($prep..., CuArray(input.weightsCdf))
        suite["gpu"][case_short]["faceBased"]["HardCodedUpwind"] = @benchmarkable gpu_HardcodedFaceBasedAssemblyRunner($prep..., "Upwind")
        suite["gpu"][case_short]["faceBased"]["HardCodedCDF"] = @benchmarkable gpu_HardcodedFaceBasedAssemblyRunner($prep..., "CDF")
        suite["gpu"][case_short]["faceBased"]["DynamicCDF"] = @benchmarkable gpu_DynamicFaceBasedAssemblyRunner($prep..., centralDifferencing)
        suite["gpu"][case_short]["faceBased"]["DynamicUpwind"] = @benchmarkable gpu_DynamicFaceBasedAssemblyRunner($prep..., upwind)
        
        # suite["gpu"][case_short]["cellBased"] = BenchmarkGroup(["gpu", "cellBased", case_long])
        # suite["gpu"][case_short]["cellBased"]["LaplaceOnly"] = @benchmarkable LaplaceOnlyCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["DivOnlyPrecalculatedWeightsUpwind"] = @benchmarkable DivOnlyPrecalculatedWeightsUpwindCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["DivOnlyPrecalculatedWeightsCDF"] = @benchmarkable DivOnlyPrecalculatedWeightsCDFCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["DivOnlyHardCodedUpwind"] = @benchmarkable DivOnlyHardcodedUpwindCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["DivOnlyHardCodedCDF"] = @benchmarkable DivOnlyHardcodedCDFCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["DivOnlyDynamicCDF"] = @benchmarkable DivOnlyDynamicCDFCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["DivOnlyDynamicUpwind"] = @benchmarkable DivOnlyDynamicUpwindCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["PrecalculatedWeightsUpwind"] = @benchmarkable PrecalculatedWeightsUpwindCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["PrecalculatedWeightsCDF"] = @benchmarkable PrecalculatedWeightsCDFCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["HardCodedUpwind"] = @benchmarkable HardcodedUpwindCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["HardCodedCDF"] = @benchmarkable HardcodedCDFCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["DynamicCDF"] = @benchmarkable DynamicCDFCellBasedAssembly($input)
        # suite["gpu"][case_short]["cellBased"]["DynamicUpwind"] = @benchmarkable DynamicUpwindCellBasedAssembly($input)
        # results = run(suite, verbose = true)
        # processResults(results)
    end
    return suite
end

function count_benchmarks(group)
    total = 0
    for v in values(group)
        if v isa BenchmarkGroup
            total += count_benchmarks(v)
        else
            total += 1
        end
    end
    return total
end

function processResults(results::BenchmarkGroup)
    # open("results.csv", "a") do io
    #     write(io, join("time_mean_ms,time_median_ms,gc_time_mean_ms,gc_time_median_ms,case_short,case_long,strategy,variant,language\n"))
    # end
    for (cpu_gpu, perDataset) in results
        println("[CPU, GPU]: $cpu_gpu ")
        for (case, perStrategy) in perDataset
            println("\t[CASE]: $case ")
            for (strategy, variant) in perStrategy
                println("\t\t[STRATEGY]: $strategy ")
                for (variation, values) in variant
                    r = Result(
                        mean(values).time / 1e6,
                        median(values).time / 1e6,
                        mean(values).gctime / 1e6,
                        median(values).gctime / 1e6,
                        case,
                        case,
                        strategy,
                        variation,
                        "julia"
                    )
                    println("\t\t\t[VARIATION]: $variation ")
                    println("\t\t\t --> time_mean_ms,time_median_ms,gc_time_mean_ms,gc_time_median_ms,case_short,case_long,strategy,variant,language")
                    println("\t\t\t --> $(ResultToCsvRow(r, false))\n")
                    open("results/results_new.csv", "a") do io
                        write(io, ResultToCsvRow(r, false))
                    end
                end
            end
        end
    end
end