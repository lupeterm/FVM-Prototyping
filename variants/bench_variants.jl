include("face_variants.jl")
include("globalface_variants.jl")
include("cell_variants.jl")
include("batched_variants.jl")
include("../cpu_threaded.jl")
function benchmark_case(case::String)
    suite = BenchmarkGroup()
    suite["cpu"] = BenchmarkGroup(["cpu"])
    meshInput = ProcessCase(Float64, case)
    input = toSOAInput(meshInput)
    suite["cpu"][case] = BenchmarkGroup(["cpu", case])
    nthreads = Threads.nthreads()
    suite["cpu"][case]["faceBased"] = BenchmarkGroup(["cpu", "faceBased", case])
    suite["cpu"][case]["cellBased"] = BenchmarkGroup(["cpu", "cellBased", case])
    suite["cpu"][case]["globalFaceBased"] = BenchmarkGroup(["cpu", "globalFaceBased", case])
    prep = prepf(meshInput)
    pde = Div{Float64,upwind{Float64}}(upwind{Float64}(), 1.0) + Laplace(1.0)
    
    if nthreads > 1
        suite["cpu"][case]["batchedFace"] = BenchmarkGroup(["cpu", "batchedFace", case])
        batches = getBatches(meshInput)

        suite["cpu"][case]["batchedFace"]["Fused-$(nthreads)"] = @benchmarkable FusedBatchedFaceBasedAssembly($input, $prep..., $batches, $pde)
        suite["cpu"][case]["batchedFace"]["PrecalculatedWeightsUpwind-$(nthreads)"] = @benchmarkable PrecalculatedWeightsBatchedFaceBasedAssembly($input, $prep..., $batches, $meshInput.weightsUpwind)
        suite["cpu"][case]["batchedFace"]["PrecalculatedWeightsCDF-$(nthreads)"] = @benchmarkable PrecalculatedWeightsBatchedFaceBasedAssembly($input, $prep..., $batches, $meshInput.weightsCdf)
        suite["cpu"][case]["batchedFace"]["HardCodedUpwind-$(nthreads)"] = @benchmarkable HardCodedBatchedFaceBasedAssembly($input, $prep..., $batches, :HardCodedUpwindBatch)
        suite["cpu"][case]["batchedFace"]["HardCodedCDF-$(nthreads)"] = @benchmarkable HardCodedBatchedFaceBasedAssembly($input, $prep..., $batches, :HardCodedCDFBatch)
        suite["cpu"][case]["batchedFace"]["DynamicCDF-$(nthreads)"] = @benchmarkable DynamicBatchedFaceBasedAssembly($input, $prep..., $batches, $:cdf_f)
        suite["cpu"][case]["batchedFace"]["DynamicUpwind-$(nthreads)"] = @benchmarkable DynamicBatchedFaceBasedAssembly($input, $prep..., $batches, $:upwind_f)

        suite["cpu"][case]["faceBased"]["Fused-$(nthreads)"] = @benchmarkable FusedFaceBasedAssembly_t($input, $prep..., $pde)
        suite["cpu"][case]["faceBased"]["PrecalculatedWeightsUpwind-$(nthreads)"] = @benchmarkable PrecalculatedWeightsFaceBasedAssembly_t($input, $prep..., $meshInput.weightsUpwind)
        suite["cpu"][case]["faceBased"]["PrecalculatedWeightsCDF-$(nthreads)"] = @benchmarkable PrecalculatedWeightsFaceBasedAssembly_t($input, $prep..., $meshInput.weightsCdf)
        suite["cpu"][case]["faceBased"]["HardCodedUpwind-$(nthreads)"] = @benchmarkable HardcodedUpwindFaceBasedAssembly_t($input, $prep...)
        suite["cpu"][case]["faceBased"]["HardCodedCDF-$(nthreads)"] = @benchmarkable HardcodedCDFFaceBasedAssembly_t($input, $prep...)
        suite["cpu"][case]["faceBased"]["DynamicCDF-$(nthreads)"] = @benchmarkable DynamicFaceBasedAssembly_t($input, $prep..., $cdf_f)
        suite["cpu"][case]["faceBased"]["DynamicUpwind-$(nthreads)"] = @benchmarkable DynamicFaceBasedAssembly_t($input, $prep..., $upwind_f)

        suite["cpu"][case]["cellBased"]["Fused-$(nthreads)"] = @benchmarkable FusedCellBasedAssembly_t($input, $prep..., $pde)
        suite["cpu"][case]["cellBased"]["PrecalculatedWeightsUpwind-$(nthreads)"] = @benchmarkable PrecalculatedWeightsCellBasedAssembly_t($input, $prep..., $meshInput.weightsUpwind)
        suite["cpu"][case]["cellBased"]["PrecalculatedWeightsCDF-$(nthreads)"] = @benchmarkable PrecalculatedWeightsCellBasedAssembly_t($input, $prep..., $meshInput.weightsCdf)
        suite["cpu"][case]["cellBased"]["HardCodedUpwind-$(nthreads)"] = @benchmarkable HardcodedUpwindCellBasedAssembly_t($input, $prep...)
        suite["cpu"][case]["cellBased"]["HardCodedCDF-$(nthreads)"] = @benchmarkable HardcodedCDFCellBasedAssembly_t($input, $prep...)
        suite["cpu"][case]["cellBased"]["DynamicCDF-$(nthreads)"] = @benchmarkable DynamicCellBasedAssembly_t($input, $prep..., $cdf_f)
        suite["cpu"][case]["cellBased"]["DynamicUpwind-$(nthreads)"] = @benchmarkable DynamicCellBasedAssembly_t($input, $prep..., $upwind_f)

        suite["cpu"][case]["globalFaceBased"]["Fused-$(nthreads)"] = @benchmarkable FusedGlobalFaceBasedAssembly_t($input, $prep..., $pde)
        suite["cpu"][case]["globalFaceBased"]["PrecalculatedWeightsUpwind-$(nthreads)"] = @benchmarkable PrecalculatedWeightsGlobalFaceBasedAssembly_t($input, $prep..., $meshInput.weightsUpwind)
        suite["cpu"][case]["globalFaceBased"]["PrecalculatedWeightsCDF-$(nthreads)"] = @benchmarkable PrecalculatedWeightsGlobalFaceBasedAssembly_t($input, $prep..., $meshInput.weightsCdf)
        suite["cpu"][case]["globalFaceBased"]["HardCodedUpwind-$(nthreads)"] = @benchmarkable HardcodedUpwindGlobalFaceBasedAssembly_t($input, $prep...)
        suite["cpu"][case]["globalFaceBased"]["HardCodedCDF-$(nthreads)"] = @benchmarkable HardcodedCDFGlobalFaceBasedAssembly_t($input, $prep...)
        suite["cpu"][case]["globalFaceBased"]["DynamicCDF-$(nthreads)"] = @benchmarkable DynamicGlobalFaceBasedAssembly_t($input, $prep..., $cdf_f)
        suite["cpu"][case]["globalFaceBased"]["DynamicUpwind-$(nthreads)"] = @benchmarkable DynamicGlobalFaceBasedAssembly_t($input, $prep..., $upwind_f)
    else
        # suite["cpu"][case]["faceBased"]["Fused-1"] = @benchmarkable FusedFaceBasedAssembly($input, $prep..., $pde)
        # suite["cpu"][case]["faceBased"]["PrecalculatedWeightsUpwind-1"] = @benchmarkable PrecalculatedWeightsFaceBasedAssembly($input, $prep..., $meshInput.weightsUpwind)
        # suite["cpu"][case]["faceBased"]["PrecalculatedWeightsCDF-1"] = @benchmarkable PrecalculatedWeightsFaceBasedAssembly($input, $prep..., $meshInput.weightsCdf)
        # suite["cpu"][case]["faceBased"]["HardCodedUpwind-1"] = @benchmarkable HardcodedUpwindFaceBasedAssembly($input, $prep...)
        # suite["cpu"][case]["faceBased"]["HardCodedCDF-1"] = @benchmarkable HardcodedCDFFaceBasedAssembly($input, $prep...)
        # suite["cpu"][case]["faceBased"]["DynamicCDF-1"] = @benchmarkable DynamicFaceBasedAssembly($input, $prep..., $cdf_f)
        # suite["cpu"][case]["faceBased"]["DynamicUpwind-1"] = @benchmarkable DynamicFaceBasedAssembly($input, $prep..., $upwind_f)

        suite["cpu"][case]["cellBased"]["Fused-1"] = @benchmarkable FusedCellBasedAssembly($input, $prep..., $pde)
        suite["cpu"][case]["cellBased"]["PrecalculatedWeightsUpwind-1"] = @benchmarkable PrecalculatedWeightsCellBasedAssembly($input, $prep..., $meshInput.weightsUpwind)
        suite["cpu"][case]["cellBased"]["PrecalculatedWeightsCDF-1"] = @benchmarkable PrecalculatedWeightsCellBasedAssembly($input, $prep..., $meshInput.weightsCdf)
        suite["cpu"][case]["cellBased"]["HardCodedUpwind-1"] = @benchmarkable HardcodedUpwindCellBasedAssembly($input, $prep...)
        suite["cpu"][case]["cellBased"]["HardCodedCDF-1"] = @benchmarkable HardcodedCDFCellBasedAssembly($input, $prep...)
        suite["cpu"][case]["cellBased"]["DynamicCDF-1"] = @benchmarkable DynamicCellBasedAssembly($input, $prep..., $cdf_f)
        suite["cpu"][case]["cellBased"]["DynamicUpwind-1"] = @benchmarkable DynamicCellBasedAssembly($input, $prep..., $upwind_f)

        # suite["cpu"][case]["globalFaceBased"]["Fused-1"] = @benchmarkable FusedGlobalFaceBasedAssembly($input, $prep..., $pde)
        # suite["cpu"][case]["globalFaceBased"]["PrecalculatedWeightsUpwind-1"] = @benchmarkable PrecalculatedWeightsGlobalFaceBasedAssembly($input, $prep..., $meshInput.weightsUpwind)
        # suite["cpu"][case]["globalFaceBased"]["PrecalculatedWeightsCDF-1"] = @benchmarkable PrecalculatedWeightsGlobalFaceBasedAssembly($input, $prep..., $meshInput.weightsCdf)
        # suite["cpu"][case]["globalFaceBased"]["HardCodedUpwind-1"] = @benchmarkable HardcodedUpwindGlobalFaceBasedAssembly($input, $prep...)
        # suite["cpu"][case]["globalFaceBased"]["HardCodedCDF-1"] = @benchmarkable HardcodedCDFGlobalFaceBasedAssembly($input, $prep...)
        # suite["cpu"][case]["globalFaceBased"]["DynamicCDF-1"] = @benchmarkable DynamicGlobalFaceBasedAssembly($input, $prep..., $cdf_f)
        # suite["cpu"][case]["globalFaceBased"]["DynamicUpwind-1"] = @benchmarkable DynamicGlobalFaceBasedAssembly($input, $prep..., $upwind_f)
    end
    results = run(suite, verbose=true)
    processResults(results, "testresults.csv", Float64)
    
end
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

function processResults(results::BenchmarkGroup, file::String, T)
    open("$(file)", "a") do io
        write(io, join("time_mean_ms,time_median_ms,gc_time_mean_ms,gc_time_median_ms,case,case_long,strategy,variant,language,precision,executor,use_kernelAbstractions,use_fusing\n"))
    end
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
                    println("\t\t\t --> time_mean_ms,time_median_ms,gc_time_mean_ms,gc_time_median_ms,case,case_long,strategy,variant,language,precision,executor,use_kernelAbstractions,use_fusing")
                    println("\t\t\t --> $(ResultToCsvRow(r, cpu_gpu == "cpu", "float64", contains(variation,"Abstract") || contains(strategy, "Abstract"), contains(variation, "Fused")))\n")
                    open("$(file)", "a") do io
                        # write(io, ResultToCsvRow(r, cpu_gpu == "cpu", "float64", contains(variation,"Abstract") || contains(strategy, "Abstract"), contains(variation, "Fused")))
                        write(io, ResultToCsvRow(r, cpu_gpu == "cpu", "float64", contains(variation, "Abstract") || true, true))
                    end
                end
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    case = ARGS[1]
    benchmark_case(case)
end