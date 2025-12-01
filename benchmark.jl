# using HeatInduction
# using MatrixAssembly
using BenchmarkTools
include("alles.jl")
using Profile
using ProfileView
testcases = [
    # ("/home/peter/clones/FVM-CFD-prototype/cases/Mycavity", "20x20"),
    # ("/home/peter/clones/FVM-CFD-prototype/cases/heat-conduction/2D-heat-conduction-on-a-2-by-2-mesh", "2x2"),
    # ("/home/peter/clones/FVM-CFD-prototype/cases/heat-conduction/2D-heat-conduction-on-a-3-by-3-mesh", "3x3"),
    # ("/home/peter/clones/FVM-CFD-prototype/cases/heat-conduction/2D-heat-conduction-on-a-10-by-10-mesh", "10x10"),
    ("/home/peter/clones/cases/ldc/Lid_driven_cavity-3d/S", "1030301 x 1030301", "Lid-Driven Cavity"),
    ("/home/peter/clones/cases/ldc2/Lid_driven_cavity-3d/M", "8000000 x 8000000", "Lid-Driven Cavity"),
    ("/home/peter/clones/cases/windsor/workspace/0172cca15b84871d59583e93ed420b34/case", "6517376x6517376", "WindsorBody" )
]

assemblyMethods = [
    # (MatrixAssembly.cellBasedAssembly, "Cell-Based Matrix Assembly"),
    # (MatrixAssembly.LdcCellBasedAssemblySparseMatrix, "Generic Cell-Based Matrix Assembly"),
    # (MatrixAssembly.cellBasedAssemblySparseMatrix, "Cell-Based Sparse Matrix Assembly"),
    # (MatrixAssembly.cellBasedAssemblySparseMultiVectorPrealloc, "Cell-Based Preallocated MultiVector Sparse Matrix Assembly"),
    # (MatrixAssembly.cellBasedAssemblySparseMultiVectorPush, "Cell-Based MultiVector Push Sparse Matrix Assembly"),
    # (MatrixAssembly.faceBasedAssembly, "Face-Based Matrix Assembly"),
    # (MatrixAssembly.batchedFaceBasedAssembly, "Batched Face-Based Matrix Assembly"),
    # (MatrixAssembly.batchedFaceBasedAssemblySparseMatrix, "Batched Face-Based Sparse Matrix Assembly")
]

for (testcase, desc, caseName) in testcases
    println("\n\n###### $caseName on a $desc mesh ######\n")
    global inputData = LidDrivenCavity(testcase)
    if typeof(inputData.U[2].values[1]) == MVector{3, Float32}
        println("\n\n--> Using VectorAssembly <--\n")
        results = @benchmark begin
            global matrices, RHSs = VectorAssembly(inputData) 
        end
        display(results)
        println()
    else
        println("\n\n--> Using Scalar <--\n")
        results = @benchmark begin
            global matrices, RHSs = ScalarAssembly(inputData) 
        end
        display(results)
        println()

    end
    println("\n\n##############################################################################################")
end