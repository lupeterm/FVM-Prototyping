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
    for (iElement, theElement) in enumerate(mesh.cells)
        RHS[iElement] = source[iElement] * theElement.volume
        diag = 0.0
        for (iFace, iFaceIndex) in enumerate(theElement.iFaces)
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor != -1
                fluxCn = diffusionCoeff[iFaceIndex] * theFace.gDiff
                fluxFn = -fluxCn
                coeffMatrix[iElement, theElement.iNeighbors[iFace]] = fluxFn
                diag += fluxCn
            else
                iBoundary = mesh.faces[iFaceIndex].patchIndex
                boundaryType = boundaryFields[iBoundary].type
                if boundaryType == "fixedValue"
                    fluxCb = diffusionCoeff[iFaceIndex] * theFace.gDiff
                    relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                    fluxVb = -fluxCb * boundaryFields[iBoundary].values[relativeFaceIndex]
                    diag += fluxCb
                    RHS[iElement] -= fluxVb
                end
            end
        end
        coeffMatrix[iElement, iElement] = diag
    end
    return coeffMatrix, RHS
end # function cellBasedAssembly

function faceBasedAssembly(input::MatrixAssemblyInput)::Tuple{Matrix{Float64},Vector{Float64}}
    mesh = input.mesh
    diffusionCoeff = input.diffusionCoeff
    boundaryFields = input.boundaryFields
    nCells = size(mesh.cells)[1]
    RHS = zeros(nCells)
    coeffMatrix = Matrix(zeros(nCells, nCells))
    for (iFace, theFace) in enumerate(mesh.faces)
        fluxC = diffusionCoeff[iFace] * theFace.gDiff
        if theFace.iNeighbor != -1
            fluxFn = -fluxC
            coeffMatrix[theFace.iOwner, theFace.iNeighbor] = fluxFn
            coeffMatrix[theFace.iNeighbor, theFace.iOwner] = fluxFn
            coeffMatrix[theFace.iOwner, theFace.iOwner] += fluxC
            coeffMatrix[theFace.iNeighbor, theFace.iNeighbor] += fluxC
        else 
            iBoundary = mesh.faces[iFace].patchIndex
            boundaryType = boundaryFields[iBoundary].type
            if boundaryType == "fixedValue"
                relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
                fluxVb = -fluxC * boundaryFields[iBoundary].values[relativeFaceIndex]
                coeffMatrix[theFace.iOwner, theFace.iOwner] += fluxC
                RHS[theFace.iOwner] -= fluxVb
            end
        end
    end
    return coeffMatrix, RHS
end # function faceBasedAssembly


function batchedFaceBasedAssembly(input::MatrixAssemblyInput)::Tuple{Matrix{Float64},Vector{Float64}}
    mesh = input.mesh
    diffusionCoeff = input.diffusionCoeff
    boundaryFields = input.boundaryFields
    nCells = size(mesh.cells)[1]
    RHS = zeros(nCells)
    coeffMatrix = Matrix(zeros(nCells, nCells))
    nInteriorFaces = mesh.numInteriorFaces
    for (iFace, theFace) in enumerate(mesh.faces[1:nInteriorFaces])
        fluxCn = diffusionCoeff[iFace] * theFace.gDiff
        fluxFn = -fluxCn
        coeffMatrix[theFace.iOwner, theFace.iNeighbor] = fluxFn
        coeffMatrix[theFace.iNeighbor, theFace.iOwner] = fluxFn

        coeffMatrix[theFace.iOwner, theFace.iOwner] += fluxCn
        coeffMatrix[theFace.iNeighbor, theFace.iNeighbor] += fluxCn
    end
    for (iBoundary, theBoundary) in enumerate(mesh.boundaries)
        startFace = theBoundary.startFace+1
        endFace = startFace + theBoundary.nFaces
        if boundaryFields[iBoundary].type == "fixedValue"
            for iFace in startFace:endFace-1
                theFace = mesh.faces[iFace]
                fluxCb = diffusionCoeff[iFace] * theFace.gDiff
                relativeFaceIndex = iFace - mesh.boundaries[theFace.patchIndex].startFace
                fluxVb = -fluxCb * boundaryFields[iBoundary].values[relativeFaceIndex]
                RHS[theFace.iOwner] -= fluxVb
                coeffMatrix[theFace.iOwner, theFace.iOwner] += fluxCb
            end
        end
    end
    return coeffMatrix, RHS
end # function batchedFaceBasedAssembly

export cellBasedAssembly, faceBasedAssembly, batchedFaceBasedAssembly
end # module MatrixAssembly
