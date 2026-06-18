include("face_variants.jl")
include("globalface_variants.jl")
include("cell_variants.jl")
include("batched_variants.jl")
include("../cpu_threaded.jl")
using ThreadPinning

function benchmark_case(case::String)
    suite = BenchmarkGroup()
    suite["cpu"] = BenchmarkGroup(["cpu"])
    meshInput = ProcessCase(Float64, case)
	numCells = meshInput.mesh.numCells
	case = "LDC-$numCells"    
	input = toSOAInput(meshInput)
    suite["cpu"][case] = BenchmarkGroup(["cpu"])
    nthreads = Threads.nthreads()
    suite["cpu"][case]["faceBased"] = BenchmarkGroup(["cpu", "faceBased"])
    suite["cpu"][case]["cellBased"] = BenchmarkGroup(["cpu", "cellBased"])
    suite["cpu"][case]["globalFaceBased"] = BenchmarkGroup(["cpu", "globalFaceBased"])
    prep = prepf(meshInput)
    pde = Div{Float64,upwind{Float64}}(upwind{Float64}(), 1.0) + Laplace(1.0)
    
    suite["cpu"][case]["batchedFace"] = BenchmarkGroup(["cpu", "batchedFace"])
    batches = getBatches(meshInput)

    suite["cpu"][case]["batchedFace"]["Fused"] = @benchmarkable FusedBatchedFaceBasedAssembly($input, $prep..., $batches, $pde)
    suite["cpu"][case]["batchedFace"]["PrecalculatedWeightsUpwind"] = @benchmarkable PrecalculatedWeightsBatchedFaceBasedAssembly($input, $prep..., $batches, $meshInput.weightsUpwind)
    suite["cpu"][case]["batchedFace"]["PrecalculatedWeightsCDF"] = @benchmarkable PrecalculatedWeightsBatchedFaceBasedAssembly($input, $prep..., $batches, $meshInput.weightsCdf)
    suite["cpu"][case]["batchedFace"]["HardCodedUpwind"] = @benchmarkable HardCodedBatchedFaceBasedAssembly($input, $prep..., $batches, :HardCodedUpwindBatch)
    suite["cpu"][case]["batchedFace"]["HardCodedCDF"] = @benchmarkable HardCodedBatchedFaceBasedAssembly($input, $prep..., $batches, :HardCodedCDFBatch)
    suite["cpu"][case]["batchedFace"]["DynamicCDF"] = @benchmarkable DynamicBatchedFaceBasedAssembly($input, $prep..., $batches, $:cdf_f)
    suite["cpu"][case]["batchedFace"]["DynamicUpwind"] = @benchmarkable DynamicBatchedFaceBasedAssembly($input, $prep..., $batches, $:upwind_f)
    results = run(suite, verbose=true)
    processResults(results, "variations_cpu_polyester_pinnedthreads.csv", Float64)
    
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
