module MatrixAssembly
using MeshStructs

function cellBasedAssembly(input::MatrixAssemblyInput)::Tuple{Matrix{Float64},Vector{Float64}}
    mesh = input.mesh
    source = input.source
    diffusionCoeff = input.diffusionCoeff
    boundaryFields = input.boundaryFields
    nCells = size(mesh.cells)[1]
    RHS = zeros(nCells)
    coeffMatrix = Matrix(zeros(nCells, nCells))
    for iElement in 1:nCells
        theElement = mesh.cells[iElement]
        RHS[iElement] = source[iElement] * theElement.volume
        diag = 0.0
        nFaces = size(theElement.iFaces)[1]
        for iFace in 1:nFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor != -1
                fluxCn = 0.0
                fluxFn = 0.0
                fluxCn = diffusionCoeff[iFaceIndex] * theFace.gDiff
                fluxFn = -fluxCn
                coeffMatrix[iElement, theElement.iNeighbors[iFace]] = fluxFn
                diag += fluxCn
            else
                iBoundary = mesh.faces[iFaceIndex].patchIndex
                boundaryType = boundaryFields[iBoundary].type
                fluxCn = 0.0
                fluxFn = 0.0
                if boundaryType == "fixedValue"
                    fluxCb = diffusionCoeff[iFaceIndex] * theFace.gDiff
                    relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                    fluxVb = -fluxCb * boundaryFields[iBoundary].values[relativeFaceIndex]
                    diag += fluxCb
                    RHS[iElement] -= fluxVb
                elseif boundaryType == "zeroGradient"
                end
            end
        end
        coeffMatrix[iElement, iElement] = diag
    end
    return coeffMatrix, RHS
end # function cellBasedAssembly

export cellBasedAssembly
end # module MatrixAssembly
