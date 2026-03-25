include("init.jl")
include("faceBased.jl")
include("globalFaceBased.jl")
include("cellBased.jl")
include("gpu_abstract.jl")
include("cpu_fused.jl")
include("gpu_faceBased.jl")
include("gpu_batchedFace.jl")
include("gpu_cellBased.jl")
include("operators.jl")
using BenchmarkTools
const CASES = [
    # ("cases/LDC-S/", "LDC-S", "Lid-Driven-Cavity S")
    # ("cases/Wind", "Wind", "WindsorBody"),
    ("cases/LDC-M", "LDC-M", "Lid-Driven-Cavity M")
]


struct Result
    time_mean_ms::Float64
    time_median_ms::Float64
    gc_time_mean_ms::Float64
    gc_time_median_ms::Float64
    case_short::String
    case_long::String
    strategy::String
    variant::String
    language::String
end

ResultToCsvRow(r::Result, cpu::Bool, precision::String, use_kernelAbstractions::Bool, use_fusing::Bool) = "$(r.time_mean_ms),$(r.time_median_ms),$(r.gc_time_mean_ms),$(r.gc_time_median_ms),$(r.case_short),$(r.case_long),$(r.strategy),$(r.variant),$(r.language),$precision,$(ifelse(cpu, "cpu", "gpu" )),$use_kernelAbstractions,$use_fusing\n"

function bench_cpu(i=nothing)
    suite = BenchmarkGroup()
    suite["cpu"] = BenchmarkGroup(["cpu"])
    suite["gpu"] = BenchmarkGroup(["gpu"])
    for (case_path, case_short, case_long) in CASES
        println("loading $case_short")
        input = if !isnothing(i)
            i
        else
            ProcessCase(Float64, case_path)
        end
        prep = prepf(input)
        pde = Div{Float64,UpwindScheme{Float64}}(UpwindScheme{Float64}(), 1.0) + Laplace(1.0)

        suite["cpu"][case_short] = BenchmarkGroup(["cpu", case_short])
        suite["cpu"][case_short]["faceBased"] = BenchmarkGroup(["cpu", "faceBased", case_long])
        suite["cpu"][case_short]["faceBased"]["FusedDivLap"] = @benchmarkable FusedFaceBasedAssembly($input, $prep...,$pde)
        suite["cpu"][case_short]["faceBased"]["LaplaceOnly"] = @benchmarkable LaplaceOnlyFaceBasedAssembly($input, $prep...)
        suite["cpu"][case_short]["faceBased"]["DivOnlyPrecalculatedWeightsUpwind"] = @benchmarkable DivOnlyPrecalculatedWeightsUpwindFaceBasedAssembly($input, $prep...)
        suite["cpu"][case_short]["faceBased"]["DivOnlyPrecalculatedWeightsCDF"] = @benchmarkable DivOnlyPrecalculatedWeightsCDFFaceBasedAssembly($input, $prep...)
        suite["cpu"][case_short]["faceBased"]["DivOnlyHardCodedUpwind"] = @benchmarkable DivOnlyHardcodedUpwindFaceBasedAssembly($input, $prep...)
        suite["cpu"][case_short]["faceBased"]["DivOnlyHardCodedCDF"] = @benchmarkable DivOnlyHardcodedCDFFaceBasedAssembly($input, $prep...)
        suite["cpu"][case_short]["faceBased"]["DivOnlyDynamicCDF"] = @benchmarkable DivOnlyDynamicCDFFaceBasedAssembly($input, $prep..., $centralDifferencing)
        suite["cpu"][case_short]["faceBased"]["DivOnlyDynamicUpwind"] = @benchmarkable DivOnlyDynamicUpwindFaceBasedAssembly($input, $prep..., $upwind)
        suite["cpu"][case_short]["faceBased"]["PrecalculatedWeightsUpwind"] = @benchmarkable PrecalculatedWeightsUpwindFaceBasedAssembly($input, $prep...)
        suite["cpu"][case_short]["faceBased"]["PrecalculatedWeightsCDF"] = @benchmarkable PrecalculatedWeightsCDFFaceBasedAssembly($input, $prep...)
        suite["cpu"][case_short]["faceBased"]["HardCodedUpwind"] = @benchmarkable HardcodedUpwindFaceBasedAssembly($input, $prep...)
        suite["cpu"][case_short]["faceBased"]["HardCodedCDF"] = @benchmarkable HardcodedCDFFaceBasedAssembly($input, $prep...)
        suite["cpu"][case_short]["faceBased"]["DynamicCDF"] = @benchmarkable DynamicCDFFaceBasedAssembly($input, $prep..., $centralDifferencing)
        suite["cpu"][case_short]["faceBased"]["DynamicUpwind"] = @benchmarkable DynamicUpwindFaceBasedAssembly($input, $prep..., $upwind)

        suite["cpu"][case_short]["cellBased"] = BenchmarkGroup(["cpu", "cellBased", case_long])
        suite["cpu"][case_short]["cellBased"]["FusedDivLap"] = @benchmarkable FusedCellBasedAssembly($input, $prep...,$pde)
        suite["cpu"][case_short]["cellBased"]["LaplaceOnly"] = @benchmarkable LaplaceOnlyCellBasedAssembly($input, $prep...)
        suite["cpu"][case_short]["cellBased"]["PrecalculatedWeightsUpwind"] = @benchmarkable PrecalculatedWeightsUpwindCellBasedAssembly($input, $prep...)
        suite["cpu"][case_short]["cellBased"]["PrecalculatedWeightsCDF"] = @benchmarkable PrecalculatedWeightsCDFCellBasedAssembly($input, $prep...)
        suite["cpu"][case_short]["cellBased"]["DynamicCDF"] = @benchmarkable DynamicCDFCellBasedAssembly($input, $prep..., $centralDifferencing)
        suite["cpu"][case_short]["cellBased"]["DynamicUpwind"] = @benchmarkable DynamicUpwindCellBasedAssembly($input, $prep..., $upwind)
        suite["cpu"][case_short]["cellBased"]["DivOnlyPrecalculatedWeightsUpwind"] = @benchmarkable DivOnlyPrecalculatedWeightsUpwindCellBasedAssembly($input, $prep...)
        suite["cpu"][case_short]["cellBased"]["DivOnlyPrecalculatedWeightsCDF"] = @benchmarkable DivOnlyPrecalculatedWeightsCDFCellBasedAssembly($input, $prep...)
        suite["cpu"][case_short]["cellBased"]["DivOnlyHardCodedUpwind"] = @benchmarkable DivOnlyHardcodedUpwindCellBasedAssembly($input, $prep...)
        suite["cpu"][case_short]["cellBased"]["DivOnlyHardCodedCDF"] = @benchmarkable DivOnlyHardcodedCDFCellBasedAssembly($input, $prep...)
        suite["cpu"][case_short]["cellBased"]["DivOnlyDynamicCDF"] = @benchmarkable DivOnlyDynamicCDFCellBasedAssembly($input, $prep..., $centralDifferencing)
        suite["cpu"][case_short]["cellBased"]["DivOnlyDynamicUpwind"] = @benchmarkable DivOnlyDynamicUpwindCellBasedAssembly($input, $prep..., $upwind)
        suite["cpu"][case_short]["cellBased"]["HardCodedUpwind"] = @benchmarkable HardcodedUpwindCellBasedAssembly($input, $prep...)
        suite["cpu"][case_short]["cellBased"]["HardCodedCDF"] = @benchmarkable HardcodedCDFCellBasedAssembly($input, $prep...)

        # suite["cpu"][case_short]["globalFaceBased"] = BenchmarkGroup(["cpu", "globalFaceBased", case_long])
        # suite["cpu"][case_short]["globalFaceBased"]["FusedDivLap"] = @benchmarkable FusedGlobalFaceBasedAssembly($input, $prep...,$pde)
        # suite["cpu"][case_short]["globalFaceBased"]["LaplaceOnly"] = @benchmarkable LaplaceOnlyGlobalFaceBasedAssembly($input, $prep...)
        # suite["cpu"][case_short]["globalFaceBased"]["DivOnlyPrecalculatedWeightsUpwind"] = @benchmarkable DivOnlyPrecalculatedWeightsUpwindGlobalFaceBasedAssembly($input, $prep...)
        # suite["cpu"][case_short]["globalFaceBased"]["DivOnlyPrecalculatedWeightsCDF"] = @benchmarkable DivOnlyPrecalculatedWeightsCDFGlobalFaceBasedAssembly($input, $prep...)
        # suite["cpu"][case_short]["globalFaceBased"]["DivOnlyHardCodedUpwind"] = @benchmarkable DivOnlyHardcodedUpwindGlobalFaceBasedAssembly($input, $prep...)
        # suite["cpu"][case_short]["globalFaceBased"]["DivOnlyHardCodedCDF"] = @benchmarkable DivOnlyHardcodedCDFGlobalFaceBasedAssembly($input, $prep...)
        # suite["cpu"][case_short]["globalFaceBased"]["DivOnlyDynamicCDF"] = @benchmarkable DivOnlyDynamicCDFGlobalFaceBasedAssembly($input, $prep..., $centralDifferencing)
        # suite["cpu"][case_short]["globalFaceBased"]["DivOnlyDynamicUpwind"] = @benchmarkable DivOnlyDynamicUpwindGlobalFaceBasedAssembly($input, $prep..., $upwind)
        # suite["cpu"][case_short]["globalFaceBased"]["PrecalculatedWeightsUpwind"] = @benchmarkable PrecalculatedWeightsUpwindGlobalFaceBasedAssembly($input, $prep...)
        # suite["cpu"][case_short]["globalFaceBased"]["PrecalculatedWeightsCDF"] = @benchmarkable PrecalculatedWeightsCDFGlobalFaceBasedAssembly($input, $prep...)
        # suite["cpu"][case_short]["globalFaceBased"]["HardCodedUpwind"] = @benchmarkable HardcodedUpwindGlobalFaceBasedAssembly($input, $prep...)
        # suite["cpu"][case_short]["globalFaceBased"]["HardCodedCDF"] = @benchmarkable HardcodedCDFGlobalFaceBasedAssembly($input, $prep...)
        # suite["cpu"][case_short]["globalFaceBased"]["DynamicCDF"] = @benchmarkable DynamicCDFGlobalFaceBasedAssembly($input, $prep..., $centralDifferencing)
        # suite["cpu"][case_short]["globalFaceBased"]["DynamicUpwind"] = @benchmarkable DynamicUpwindGlobalFaceBasedAssembly($input, $prep..., $upwind)

        results = run(suite, verbose=true)
        processResults(results, "results_f64.csv", Float64)
        GC.gc()
    end
    # return suite
end



function bench_fused(i=nothing)
    suite = BenchmarkGroup()
    suite["fused-cpu"] = BenchmarkGroup(["fused-cpu"])
    for (case_path, case_short, case_long) in CASES
        println("loading $case_short")
        input = if !isnothing(i)
            i
        else
            ProcessCase(case_path)
        end

        upwindDivLap::Vector{FVMOP} = [DIV(UpwindScheme(), 1.0), LAPLACE(1.0)]
        cDFDivLap::Vector{FVMOP} = [DIV(CentralDiffScheme(), 1.0), LAPLACE(1.0)]
        divOnlyUpwind::Vector{FVMOP} = [DIV(UpwindScheme(), 1.0)]
        divOnlyCDF::Vector{FVMOP} = [DIV(CentralDiffScheme(), 1.0)]
        laplaceOnly::Vector{FVMOP} = [LAPLACE(1.0)]



        suite["fused-cpu"][case_short] = BenchmarkGroup(["fused-cpu", case_short])
        suite["fused-cpu"][case_short]["faceBased"] = BenchmarkGroup(["fused-cpu", "faceBased", case_long])
        suite["fused-cpu"][case_short]["faceBased"]["UpwindDivLap"] = @benchmarkable FusedFaceBasedAssembly($input, $upwindDivLap)
        suite["fused-cpu"][case_short]["faceBased"]["CDFDivLap"] = @benchmarkable FusedFaceBasedAssembly($input, $cDFDivLap)
        suite["fused-cpu"][case_short]["faceBased"]["DivOnlyUpwind"] = @benchmarkable FusedFaceBasedAssembly($input, $divOnlyUpwind)
        suite["fused-cpu"][case_short]["faceBased"]["DivOnlyCDF"] = @benchmarkable FusedFaceBasedAssembly($input, $divOnlyCDF)
        suite["fused-cpu"][case_short]["faceBased"]["LaplaceOnly"] = @benchmarkable FusedFaceBasedAssembly($input, $laplaceOnly)

        results = run(suite, verbose=true)
        processResults(results, "results_fused.csv")
        return
    end
    # return suite
end


function bench_gpu(i=nothing, T=Float64)
    suite = BenchmarkGroup()
    for (case_path, case_short, case_long) in CASES
        suite["gpu"] = BenchmarkGroup(["gpu"])
        println("loading $case_short")
        input = if !isnothing(i)
            i
        else
            ProcessCase(T, case_path)
        end

        suite["gpu"][case_short] = BenchmarkGroup(["gpu", case_short])

        # prep = gpu_prepareBatchedFaceBased(input)
        wUp = CuArray(input.weightsUpwind)
        wCdf = CuArray(input.weightsCdf)
        # fusedBatchedPrep = getBatchedFaceBasedGpuInput(input) 

        pde = Div{Float64,UpwindScheme{Float64}}(UpwindScheme{Float64}(), 1.0) + Laplace(1.0)
        # suite["gpu"][case_short]["batchedFace"] = BenchmarkGroup(["gpu", "batchedFace", case_long])
        # suite["gpu"][case_short]["batchedFace"]["AbstractFusedDivLap"] = @benchmarkable run_batchedFace_abstract($fusedBatchedPrep..., $pde)
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyPrecalculatedWeightsUpwind"] = @benchmarkable gpu_DivOnlyPrecalculatedWeightsBatchedFaceAssemblyRunner($prep..., $wUp)
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyPrecalculatedWeightsCDF"] = @benchmarkable gpu_DivOnlyPrecalculatedWeightsBatchedFaceAssemblyRunner($prep..., $wCdf)
        # suite["gpu"][case_short]["batchedFace"]["PrecalculatedWeightsUpwind"] = @benchmarkable gpu_PrecalculatedWeightsBatchedFaceAssemblyRunner($prep..., $wUp)
        # suite["gpu"][case_short]["batchedFace"]["PrecalculatedWeightsCDF"] = @benchmarkable gpu_PrecalculatedWeightsBatchedFaceAssemblyRunner($prep..., $wCdf)
        # suite["gpu"][case_short]["batchedFace"]["LaplaceOnly"] = @benchmarkable gpu_LaplaceOnlyBatchedFaceAssemblyRunner($prep...)
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyHardCodedUpwind"] = @benchmarkable gpu_DivOnlyHardcodedDivBatchedFaceAssemblyRunner($prep..., $"Upwind")
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyHardCodedCDF"] = @benchmarkable gpu_DivOnlyHardcodedDivBatchedFaceAssemblyRunner($prep..., $"CDF")
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyDynamicCDF"] = @benchmarkable gpu_DivOnlyDynamicBatchedFaceAssemblyRunner($prep..., $centralDifferencing)
        # suite["gpu"][case_short]["batchedFace"]["DivOnlyDynamicUpwind"] = @benchmarkable gpu_DivOnlyDynamicBatchedFaceAssemblyRunner($prep..., $upwind)
        # suite["gpu"][case_short]["batchedFace"]["HardCodedUpwind"] = @benchmarkable gpu_HardcodedBatchedFaceAssemblyRunner($prep..., $"Upwind")
        # suite["gpu"][case_short]["batchedFace"]["HardCodedCDF"] = @benchmarkable gpu_HardcodedBatchedFaceAssemblyRunner($prep..., $"CDF")
        # suite["gpu"][case_short]["batchedFace"]["DynamicCDF"] = @benchmarkable gpu_DynamicBatchedFaceAssemblyRunner($prep..., $centralDifferencing)
        # suite["gpu"][case_short]["batchedFace"]["DynamicUpwind"] = @benchmarkable gpu_DynamicBatchedFaceAssemblyRunner($prep..., $upwind)
        # prep2 = gpu_prepareFaceBased(input)
        # fusedFacePrep = getFaceBasedGpuInput(input) 
        # suite["gpu"][case_short]["faceBased"] = BenchmarkGroup(["gpu", "faceBased", case_long])
        # suite["gpu"][case_short]["faceBased"]["AbstractFusedDivLap"] = @benchmarkable run_faceBased_abstract($fusedFacePrep..., $pde)
        # suite["gpu"][case_short]["faceBased"]["DivOnlyPrecalculatedWeightsUpwind"] = @benchmarkable gpu_DivOnlyPrecalculatedWeightsFaceBasedAssemblyRunner($prep2..., $wUp)
        # suite["gpu"][case_short]["faceBased"]["DivOnlyPrecalculatedWeightsCDF"] = @benchmarkable gpu_DivOnlyPrecalculatedWeightsFaceBasedAssemblyRunner($prep2..., $wCdf)
        # suite["gpu"][case_short]["faceBased"]["PrecalculatedWeightsUpwind"] = @benchmarkable gpu_PrecalculatedWeightsFaceBasedAssemblyRunner($prep2..., $wUp)
        # suite["gpu"][case_short]["faceBased"]["PrecalculatedWeightsCDF"] = @benchmarkable gpu_PrecalculatedWeightsFaceBasedAssemblyRunner($prep2..., $wCdf)
        # suite["gpu"][case_short]["faceBased"]["LaplaceOnly"] = @benchmarkable gpu_LaplaceOnlyFaceBasedAssemblyRunner($prep2...)
        # suite["gpu"][case_short]["faceBased"]["DivOnlyHardCodedUpwind"] = @benchmarkable gpu_DivOnlyHardcodedDivFaceBasedAssemblyRunner($prep2..., $"Upwind")
        # suite["gpu"][case_short]["faceBased"]["DivOnlyHardCodedCDF"] = @benchmarkable gpu_DivOnlyHardcodedDivFaceBasedAssemblyRunner($prep2..., $"CDF")
        # suite["gpu"][case_short]["faceBased"]["DivOnlyDynamicCDF"] = @benchmarkable gpu_DivOnlyDynamicFaceBasedAssemblyRunner($prep2..., $centralDifferencing)
        # suite["gpu"][case_short]["faceBased"]["DivOnlyDynamicUpwind"] = @benchmarkable gpu_DivOnlyDynamicFaceBasedAssemblyRunner($prep2..., $upwind)
        # suite["gpu"][case_short]["faceBased"]["HardCodedUpwind"] = @benchmarkable gpu_HardcodedFaceBasedAssemblyRunner($prep2..., $"Upwind")
        # suite["gpu"][case_short]["faceBased"]["HardCodedCDF"] = @benchmarkable gpu_HardcodedFaceBasedAssemblyRunner($prep2..., $"CDF")
        # suite["gpu"][case_short]["faceBased"]["DynamicCDF"] = @benchmarkable gpu_DynamicFaceBasedAssemblyRunner($prep2..., $centralDifferencing)
        # suite["gpu"][case_short]["faceBased"]["DynamicUpwind"] = @benchmarkable gpu_DynamicFaceBasedAssemblyRunner($prep2..., $upwind)
        cellBasedPrep = getCellBasedGpuInput(input)
        suite["gpu"][case_short]["cellBased"] = BenchmarkGroup(["gpu", "cellBased", case_long])
        suite["gpu"][case_short]["cellBased"]["AbstractFusedDivLap"] = @benchmarkable runCellkernel($cellBasedPrep, $pde) 
        suite["gpu"][case_short]["cellBased"]["DivOnlyPrecalculatedWeightsUpwind"] = @benchmarkable gpu_DivOnlyPrecalculatedWeightsCellBasedRunner($cellBasedPrep, $wUp)
        suite["gpu"][case_short]["cellBased"]["DivOnlyPrecalculatedWeightsCDF"] = @benchmarkable gpu_DivOnlyPrecalculatedWeightsCellBasedRunner($cellBasedPrep, $wCdf)
        suite["gpu"][case_short]["cellBased"]["PrecalculatedWeightsUpwind"] = @benchmarkable gpu_PrecalculatedWeightsCellBasedRunner($cellBasedPrep, $wUp)
        suite["gpu"][case_short]["cellBased"]["PrecalculatedWeightsCDF"] = @benchmarkable gpu_PrecalculatedWeightsCellBasedRunner($cellBasedPrep, $wCdf)
        suite["gpu"][case_short]["cellBased"]["LaplaceOnly"] = @benchmarkable gpu_LaplaceOnlyCellBasedRunner($cellBasedPrep...)
        suite["gpu"][case_short]["cellBased"]["DivOnlyHardCodedUpwind"] = @benchmarkable gpu_DivOnlyHardcodedUpwindCellBasedRunner($cellBasedPrep...)
        suite["gpu"][case_short]["cellBased"]["DivOnlyHardCodedCDF"] = @benchmarkable gpu_DivOnlyHardcodedCDFCellBasedRunner($cellBasedPrep...)
        suite["gpu"][case_short]["cellBased"]["DivOnlyDynamicCDF"] = @benchmarkable gpu_DivOnlyDynamicCellBasedRunner($cellBasedPrep, $centralDifferencing)
        suite["gpu"][case_short]["cellBased"]["DivOnlyDynamicUpwind"] = @benchmarkable gpu_DivOnlyDynamicCellBasedRunner($cellBasedPrep, $upwind)
        suite["gpu"][case_short]["cellBased"]["HardCodedUpwind"] = @benchmarkable gpu_HardcodedUpwindCellBasedRunner($cellBasedPrep...)
        suite["gpu"][case_short]["cellBased"]["HardCodedCDF"] = @benchmarkable gpu_HardcodedCDFCellBasedRunner($cellBasedPrep...)
        suite["gpu"][case_short]["cellBased"]["DynamicCDF"] = @benchmarkable gpu_DynamicCellBasedRunner($cellBasedPrep, $centralDifferencing)
        suite["gpu"][case_short]["cellBased"]["DynamicUpwind"] = @benchmarkable gpu_DynamicCellBasedRunner($cellBasedPrep, $upwind)
    end
    results = run(suite, verbose=true)
    processResults(results, "results_f64.csv", Float64)
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

function processResults(results::BenchmarkGroup, file::String, T)
    # open("results/$(file)", "a") do io
    #     write(io, join("time_mean_ms,time_median_ms,gc_time_mean_ms,gc_time_median_ms,case_short,case_long,strategy,variant,language,precision,executor,use_kernelAbstractions,use_fusing\n"))
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
                    println("\t\t\t --> time_mean_ms,time_median_ms,gc_time_mean_ms,gc_time_median_ms,case_short,case_long,strategy,variant,language,precision,executor,use_kernelAbstractions,use_fusing")
                    println("\t\t\t --> $(ResultToCsvRow(r, cpu_gpu == "cpu", "float64", contains(variation,"Abstract") || contains(strategy, "Abstract"), contains(variation, "Fused")))\n")
                    open("results/$(file)", "a") do io
                        write(io, ResultToCsvRow(r, cpu_gpu == "cpu", "float64", contains(variation,"Abstract") || contains(strategy, "Abstract"), contains(variation, "Fused")))
                    end
                end
            end
        end
    end
end



## TODO vergleich KernelAbstractions vs JuliaGPU
function compare_KA_CUDA(i=nothing)
    suite = BenchmarkGroup()
    for (case_path, case_short, case_long) in CASES
        println("loading $case_short")
        suite[case_short] = BenchmarkGroup([case_short])

        input = if !isnothing(i)
            i
        else
            ProcessCase(case_path)
        end
        pde = Div{Float64,UpwindScheme{Float64}}(UpwindScheme{Float64}(), 1.0) + Laplace(1.0)
        kaPrep = getBatchedFaceBasedGpuInput(input)
        suite[case_short]["KernelAbstractions"] = BenchmarkGroup(["KernelAbstractions", case_short])
        suite[case_short]["KernelAbstractions"]["FusedBatchedFace"] = @benchmarkable run_batchedFace_abstract($kaPrep..., $pde)

        suite[case_short]["JuliaGPU"] = BenchmarkGroup(["JuliaGPU", case_short])
        suite[case_short]["JuliaGPU"]["FusedBatchedFace"] = @benchmarkable run_batchedFace_abstract($kaPrep..., $pde)

    end
end