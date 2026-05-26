include("gpu_face_variants.jl")
# include("gpu_globalface_variants.jl")
include("gpu_cell_variants.jl")
include("batched_variants.jl")
include("gpu_batched_variants.jl")
include("../operators.jl")

function benchmark_case(case::String)
    suite = BenchmarkGroup()
    suite["gpu"] = BenchmarkGroup(["gpu"])
    meshInput = ProcessCase(Float64, case)
	numCells = meshInput.mesh.numCells
	case = "LDC-$numCells"    
    suite["gpu"][case] = BenchmarkGroup(["gpu"])
    nthreads = Threads.nthreads()
    suite["gpu"][case]["faceBased"] = BenchmarkGroup(["gpu", "faceBased"])
    suite["gpu"][case]["cellBased"] = BenchmarkGroup(["gpu", "cellBased"])
    suite["gpu"][case]["globalFaceBased"] = BenchmarkGroup(["gpu", "globalFaceBased"])
    
    pde = Div{Float64,upwind{Float64}}(upwind{Float64}(), 1.0) + Laplace(1.0)
    
    wUp = CuArray(meshInput.weightsUpwind)
    wCdf = CuArray(meshInput.weightsCdf)
    prep = getBatchedFaceBasedGpuInput(meshInput)
    suite["gpu"][case]["batchedFace"]["Fused"] = @benchmarkable FusedBatchedAssembly($prep..., $pde)
    suite["gpu"][case]["batchedFace"]["PrecalculatedWeightsUpwind"] = @benchmarkable PrecalculatedWeightsBatchedAssembly($prep..., $wUp)
    suite["gpu"][case]["batchedFace"]["PrecalculatedWeightsCDF"] = @benchmarkable PrecalculatedWeightsBatchedAssembly($prep..., $wCdf)
    suite["gpu"][case]["batchedFace"]["HardCodedUpwind"] = @benchmarkable HardcodedBatchedAssembly($prep..., $"upwind_f")
    suite["gpu"][case]["batchedFace"]["HardCodedCDF"] = @benchmarkable HardcodedBatchedAssembly($prep..., $"CDF")
    suite["gpu"][case]["batchedFace"]["DynamicCDF"] = @benchmarkable DynamicBatchedAssembly($prep..., $cdf_f)
    suite["gpu"][case]["batchedFace"]["DynamicUpwind"] = @benchmarkable DynamicBatchedAssembly($prep..., $upwind_f)

    input_facebased = faceInput(meshInput)
    suite["gpu"][case]["faceBased"]["Fused"] = @benchmarkable gpu_fusedFaceBasedAssemblyRunner($input_facebased..., $pde)
    suite["gpu"][case]["faceBased"]["PrecalculatedWeightsUpwind"] = @benchmarkable gpu_PrecalculatedWeightsFaceBasedAssemblyRunner($input_facebased..., $wUp)
    suite["gpu"][case]["faceBased"]["PrecalculatedWeightsCDF"] = @benchmarkable gpu_PrecalculatedWeightsFaceBasedAssemblyRunner($input_facebased..., $wCdf)
    suite["gpu"][case]["faceBased"]["HardCodedUpwind"] = @benchmarkable gpu_HardcodedFaceBasedAssemblyRunner($input_facebased..., $"upwind_f")
    suite["gpu"][case]["faceBased"]["HardCodedCDF"] = @benchmarkable gpu_HardcodedFaceBasedAssemblyRunner($input_facebased..., $"CDF")
    suite["gpu"][case]["faceBased"]["DynamicCDF"] = @benchmarkable gpu_DynamicFaceBasedAssemblyRunner($input_facebased..., $cdf_f)
    suite["gpu"][case]["faceBased"]["DynamicUpwind"] = @benchmarkable gpu_DynamicFaceBasedAssemblyRunner($input_facebased..., $upwind_f)

    cellBasedPrep = getCellBasedGpuInput(meshInput)
    suite["gpu"][case]["cellBased"]["Fused"] = @benchmarkable FusedCell($cellBasedPrep, $pde) 
    suite["gpu"][case]["cellBased"]["PrecalculatedWeightsUpwind"] = @benchmarkable PrecalculatedWeightsCell($cellBasedPrep, $wUp)
    suite["gpu"][case]["cellBased"]["PrecalculatedWeightsCDF"] = @benchmarkable PrecalculatedWeightsCell($cellBasedPrep, $wCdf)
    suite["gpu"][case]["cellBased"]["HardCodedUpwind"] = @benchmarkable HardcodedCell($cellBasedPrep, $"upwind_f")
    suite["gpu"][case]["cellBased"]["HardCodedCDF"] = @benchmarkable HardcodedCell($cellBasedPrep, $"CDF")
    suite["gpu"][case]["cellBased"]["DynamicCDF"] = @benchmarkable DynamicCell($cellBasedPrep, $cdf_f)
    suite["gpu"][case]["cellBased"]["DynamicUpwind"] = @benchmarkable DynamicCell($cellBasedPrep, $upwind_f)
    results = run(suite, verbose=true)
    processResults(results, "variations_gpu.csv", Float64)
    
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

ResultToCsvRow(r::Result, cpu::Bool, precision::String, use_kernelAbstractions::Bool, use_fusing::Bool) = "$(r.time_mean_ms),$(r.time_median_ms),$(r.gc_time_mean_ms),$(r.gc_time_median_ms),$(r.case_short),$(r.case_long),$(r.strategy),$(r.variant),$(r.language),$precision,$(ifelse(cpu, "cpu", "gpu" )),$use_kernelAbstractions,$use_fusing,$(Threads.nthreads())\n"

function processResults(results::BenchmarkGroup, file::String, T)
    if !isfile(file)
		open("$(file)", "a") do io
	        write(io, join("time_mean_ms,time_median_ms,gc_time_mean_ms,gc_time_median_ms,case,case_long,strategy,variant,language,precision,executor,use_kernelAbstractions,use_fusing,Threads\n"))
	    end
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
                    println("\t\t\t --> time_mean_ms,time_median_ms,gc_time_mean_ms,gc_time_median_ms,case,case_long,strategy,variant,language,precision,executor,use_kernelAbstractions,use_fusing,Threads")
                    println("\t\t\t --> $(ResultToCsvRow(r, cpu_gpu == "cpu", "float64", contains(variation,"Abstract") || contains(strategy, "Abstract"), contains(variation, "Fused")))\n")
                    open("$(file)", "a") do io
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
