using HeatInduction
using MatrixAssembly
using BenchmarkTools
import LinearSolve as LS

testcases = [
    ("/Users/peter/clones/FVM-CFD-prototype/cases/heat-conduction/2D-heat-conduction-on-a-2-by-2-mesh", "2x2"),
    ("/Users/peter/clones/FVM-CFD-prototype/cases/heat-conduction/2D-heat-conduction-on-a-3-by-3-mesh", "3x3"),
    ("/Users/peter/clones/FVM-CFD-prototype/cases/heat-conduction/2D-heat-conduction-on-a-10-by-10-mesh", "10x10")
]

assemblyMethods = [
    (MatrixAssembly.cellBasedAssembly, "Cell-Based Matrix Assembly"),
    (MatrixAssembly.cellBasedAssemblySparseMatrix, "Cell-Based Sparse Matrix Assembly"),
    (MatrixAssembly.cellBasedAssemblySparseMultiVectorPrealloc, "Cell-Based Preallocated MultiVector Sparse Matrix Assembly"),
    (MatrixAssembly.cellBasedAssemblySparseMultiVectorPush, "Cell-Based MultiVector Push Sparse Matrix Assembly"),
    (MatrixAssembly.faceBasedAssembly, "Face-Based Matrix Assembly"),
    (MatrixAssembly.batchedFaceBasedAssembly, "Batched Face-Based Matrix Assembly"),
    (MatrixAssembly.batchedFaceBasedAssemblySparseMatrix, "Batched Face-Based Sparse Matrix Assembly")
]

for (testcase, desc) in testcases
    println("\n\n###### heat conduction on a $desc mesh ######\n")
    global inputData = HeatInduction.heatInduction(testcase)
    for (assemblyMethod, mDesc) in assemblyMethods
        println("\n--> Using $mDesc <--\n")
        results = @benchmark begin
            global matrix, RHS = $assemblyMethod(inputData) 
        end
        display(results)
        if desc != "10x10"
            display(matrix)
        end
    end
    println("\n\n##############################################################################################")
end