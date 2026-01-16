import Pkg
Pkg.activate(".")
using BenchmarkTools
include("alles.jl")
Pkg.precompile()
using Profile
using ProfileView
caseBaseFolder = "cases/"

for (_, dirs, _) in readdir(caseBaseFolder)
    for c in dirs
        println("\n\n###### $c ######\n")
        case = joinpath(caseBaseFolder, c);
        global inputData = LidDrivenCavity(case);
        @time begin
            global mats = ThreadedCellBasedAssembly(inputData);
        end
        display(mats[1])
        GC.gc()
    end
end

# for (testcase, desc, caseName) in testcases
#     println("\n\n###### $caseName on a $desc mesh ######\n")
#     global inputData = LidDrivenCavity(testcase)
#     if typeof(inputData.U[2].values[1]) == MVector{3, Float32}
#         println("\n\n--> Using VectorAssembly <--\n")
#         results = @benchmark begin
#             global matrices, RHSs = VectorAssembly(inputData) 
#         end
#         display(results)
#         println()
#     else
#         println("\n\n--> Using Scalar <--\n")
#         results = @benchmark begin
#             global matrices, RHSs = ScalarAssembly(inputData) 
#         end
#         display(results)
#         println()

#     end
#     println("\n\n##############################################################################################")
# end