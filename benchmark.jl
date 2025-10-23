using HeatInduction
using MatrixAssembly
using BenchmarkTools
import LinearSolve as LS

testcases = [
    # ("/home/peter/clones/FVM-CFD-prototype/cases/Mycavity", "20x20"),
    # ("/home/peter/clones/FVM-CFD-prototype/cases/heat-conduction/2D-heat-conduction-on-a-2-by-2-mesh", "2x2"),
    # ("/home/peter/clones/FVM-CFD-prototype/cases/heat-conduction/2D-heat-conduction-on-a-3-by-3-mesh", "3x3"),
    # ("/home/peter/clones/FVM-CFD-prototype/cases/heat-conduction/2D-heat-conduction-on-a-10-by-10-mesh", "10x10"),
    ("/home/peter/Documents/uni/openfoam-benchmarking/LidDrivenCavityS/Lid_driven_cavity-3d/S", "1030301 x 1030301", "Lid-Driven Cavity")
]

assemblyMethods = [
    # (MatrixAssembly.cellBasedAssembly, "Cell-Based Matrix Assembly"),
    (MatrixAssembly.LdcCellBasedAssemblySparseMatrix, "Generic Cell-Based Matrix Assembly"),
    # (MatrixAssembly.cellBasedAssemblySparseMatrix, "Cell-Based Sparse Matrix Assembly"),
    # (MatrixAssembly.cellBasedAssemblySparseMultiVectorPrealloc, "Cell-Based Preallocated MultiVector Sparse Matrix Assembly"),
    # (MatrixAssembly.cellBasedAssemblySparseMultiVectorPush, "Cell-Based MultiVector Push Sparse Matrix Assembly"),
    # (MatrixAssembly.faceBasedAssembly, "Face-Based Matrix Assembly"),
    # (MatrixAssembly.batchedFaceBasedAssembly, "Batched Face-Based Matrix Assembly"),
    # (MatrixAssembly.batchedFaceBasedAssemblySparseMatrix, "Batched Face-Based Sparse Matrix Assembly")
]

for (testcase, desc, caseName) in testcases
    println("\n\n###### $caseName on a $desc mesh ######\n")
    global inputData = HeatInduction.LidDrivenCavity(testcase)
    for (assemblyMethod, mDesc) in assemblyMethods
        println("\n\n--> Using $mDesc <--\n")
        # results = @benchmark begin
            global matrices, RHSs = assemblyMethod(inputData) 
        # end
        display(results)
        # if desc != "10x10"
        for matrix in matrices
            display(matrix)
        end
    end
    println("\n\n##############################################################################################")
end

# global i = 17
# global lines = readlines("/home/peter/Documents/uni/openfoam-benchmarking/LidDrivenCavityS/Lid_driven_cavity-3d/S/0/U")
# while !startswith(lines[i], "internalField")
#     global i += 1
# end
# global split18 = match(r"^(\w+)\s+(\w+)\s(?:(\d)|(\([\d\s]+\)));", lines[i])
# println(split18)
# while !contains(lines[i], "boundaryField")
#     global i += 1
# end
# boundaryLines = lines[i+2:end-4]
# joined = join(boundaryLines)
# splitted = split(joined, "}")
# for (index, b) in enumerate(splitted)
#     matches = match(r"\s*(\w+)\s*\{\s*type\s*(\w+);(?:\s*value\s*(\w+) (?:(\d+)|(\([\d\s]+\)));)?", b)
#     println(matches)
# end