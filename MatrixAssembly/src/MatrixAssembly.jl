module MatrixAssembly
using MeshStructs
using SparseArrays
using StaticArrays
using ProgressBars
function cellBasedAssembly(input::MatrixAssemblyInput)::Tuple{Matrix{Float64},Vector{Float64}}
    mesh = input.mesh
    source = input.source
    diffusionCoeff = input.diffusionCoeff # Γ
    boundaryFields = input.boundaryFields
    nCells = size(mesh.cells)[1]
    RHS = zeros(nCells)
    coeffMatrix = Matrix(zeros(nCells, nCells))
    for (iElement, theElement) in enumerate(mesh.cells)
        # bC = Q_C*V_C - bigsum(FluxVf)
        RHS[iElement] = source[iElement] * theElement.volume  # S_u
        for (iFace, iFaceIndex) in enumerate(theElement.iFaces)
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor != -1
                # FluxCn = Γ_n * gdiff_n  (p.215)
                fluxCn = diffusionCoeff[iFaceIndex] * theFace.gDiff
                fluxFn = -fluxCn
                coeffMatrix[iElement, theElement.iNeighbors[iFace]] = fluxFn
                # accumulate neighbors into diag 
                coeffMatrix[iElement, iElement] += fluxCn
            else
                iBoundary = mesh.faces[iFaceIndex].patchIndex
                boundaryType = boundaryFields[iBoundary].type
                if boundaryType == "fixedValue"
                    fluxCb = diffusionCoeff[iFaceIndex] * theFace.gDiff
                    relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                    fluxVb = -fluxCb * boundaryFields[iBoundary].values[relativeFaceIndex]
                    coeffMatrix[iElement, iElement] += fluxCb
                    RHS[iElement] -= fluxVb
                end
            end
        end
    end
    return coeffMatrix, RHS
end # function cellBasedAssembly



function cellBasedAssemblySparseMatrix(input::MatrixAssemblyInput)::Tuple{SparseMatrixCSC{Float64,Int64},SparseVector{Float64,Int64}}
    mesh = input.mesh
    source = input.source
    diffusionCoeff = input.diffusionCoeff
    boundaryFields = input.boundaryFields
    nCells = size(mesh.cells)[1]
    RHS = spzeros(nCells)
    coeffMatrix = spzeros(nCells, nCells)
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
end # function cellBasedAssemblySparseMatrix

function cellBasedAssemblySparseMultiVectorPrealloc(input::MatrixAssemblyInput)::Tuple{SparseMatrixCSC{Float64,Int64},Vector{Float64}}
    mesh = input.mesh
    source = input.source
    diffusionCoeff = input.diffusionCoeff
    boundaryFields = input.boundaryFields
    nCells = size(mesh.cells)[1]
    RHS = zeros(nCells)
    rows = zeros(nCells * nCells)
    cols = zeros(nCells * nCells)
    vals::Vector{Float64} = zeros(nCells * nCells)
    idx = 1
    for (iElement, theElement) in enumerate(mesh.cells)
        RHS[iElement] = source[iElement] * theElement.volume
        diag = 0.0
        for (iFace, iFaceIndex) in enumerate(theElement.iFaces)
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor != -1
                fluxCn = diffusionCoeff[iFaceIndex] * theFace.gDiff
                fluxFn = -fluxCn
                rows[idx] = iElement
                cols[idx] = theElement.iNeighbors[iFace]
                vals[idx] = fluxFn
                idx += 1
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
        rows[idx] = iElement
        cols[idx] = iElement
        vals[idx] = diag
        idx += 1
    end
    idx -= 1
    return SparseArrays.sparse(
        rows[1:idx],
        cols[1:idx],
        vals[1:idx],
    ), RHS
end # function cellBasedAssemblySparseMultiVectorPrealloc

function cellBasedAssemblySparseMultiVectorPush(input::MatrixAssemblyInput)::Tuple{SparseMatrixCSC{Float64,Int64},Vector{Float64}}
    mesh = input.mesh
    source = input.source
    diffusionCoeff = input.diffusionCoeff
    boundaryFields = input.boundaryFields
    nCells = size(mesh.cells)[1]
    RHS = zeros(nCells)
    rows = []
    cols = []
    vals::Vector{Float64} = []
    for (iElement, theElement) in enumerate(mesh.cells)
        RHS[iElement] = source[iElement] * theElement.volume
        diag = 0.0
        for (iFace, iFaceIndex) in enumerate(theElement.iFaces)
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor != -1
                fluxCn = diffusionCoeff[iFaceIndex] * theFace.gDiff
                fluxFn = -fluxCn
                push!(rows, iElement)
                push!(cols, theElement.iNeighbors[iFace])
                push!(vals, fluxFn)
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
        push!(rows, iElement)
        push!(cols, iElement)
        push!(vals, diag)
    end
    return SparseArrays.sparse(
        rows, cols, vals
    ), RHS
end # function cellBasedAssemblySparseMultiVectorPush

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
        startFace = theBoundary.startFace + 1
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

function batchedFaceBasedAssemblySparseMatrix(input::MatrixAssemblyInput)::Tuple{SparseMatrixCSC{Float64,Int64},SparseVector{Float64,Int64}}
    mesh = input.mesh
    diffusionCoeff = input.diffusionCoeff
    boundaryFields = input.boundaryFields
    nCells = size(mesh.cells)[1]
    RHS = spzeros(nCells)
    coeffMatrix = spzeros(nCells, nCells)
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
        startFace = theBoundary.startFace + 1
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
end # function batchedFaceBasedAssemblySparseMatrix

function genericCellBasedAssemblySparseMatrix(input::GenericMatrixAssemblyInput)::Tuple{Vector{SparseMatrixCSC{Float64,Int64}},Vector{Vector{Float64}}}
    mesh = input.mesh
    variables = input.variables
    boundaryFields = input.boundaryFields
    nCells = size(mesh.cells)[1]
    mappings = input.mappings
    numVariables = length(mappings)
    coeffMatrices = [spzeros(nCells, nCells) for _ in 1:numVariables]
    RHSs = input.sources != [] ? input.sources : [zeros(nCells) for _ in 1:numVariables]
    for (iElement, theElement) in enumerate(mesh.cells)
        for iVar in 1:numVariables
            RHSs[iVar][iElement] *= theElement.volume  # S_u
        end
        for (iFace, iFaceIndex) in enumerate(theElement.iFaces)
            theFace = mesh.faces[iFaceIndex]
            for (iVariable, variable) in enumerate(variables)
                if theFace.iNeighbor != -1
                    fluxCn = variable[iFaceIndex] * theFace.gDiff
                    fluxFn = -fluxCn
                    coeffMatrices[iVariable][iElement, theElement.iNeighbors[iFace]] = fluxFn
                    coeffMatrices[iVariable][iElement, iElement] += fluxCn
                else
                    iBoundary = mesh.faces[iFaceIndex].patchIndex
                    boundaryType = boundaryFields[iVariable][iBoundary].type
                    if boundaryType == "fixedValue"
                        fluxCb = variable[iFaceIndex] * theFace.gDiff
                        relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                        fluxVb = -fluxCb * boundaryFields[iVariable][iBoundary].values[relativeFaceIndex]
                        coeffMatrices[iVariable][iElement, iElement] += fluxCb
                        RHSs[iVariable][iElement] -= fluxVb
                    end
                end
            end
        end
    end
    return coeffMatrices, RHSs
end # function genericCellBasedAssemblySparseMatrix

function LdcCellBasedAssemblySparseMultiVectorPrealloc(input::LdcMatrixAssemblyInput)::Tuple{SparseMatrixCSC{MVector{3,Float64},Int64},SparseMatrixCSC{Float64, Int64}}
    mesh = input.mesh
    nu = input.nu
    velocity = input.U
    nCells = size(mesh.cells)[1]
    RHS = spzeros(nCells, 3)  # [ux,uy,uz]
    rows = []
    cols = []
    vals::Vector{MVector{3, Float64}} = []
    for (iElement, theElement) in enumerate(mesh.cells)
        diag = 0.0
        for (iFace, iFaceIndex) in enumerate(theElement.iFaces)
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor != -1
                fluxCn = nu * theFace.gDiff
                fluxFn = -fluxCn
                push!(rows, iElement)
                push!(cols, theElement.iNeighbors[iFace])
                push!(vals, MVector(fluxFn, fluxFn, fluxFn))
                diag += fluxCn
            else
                iBoundary = mesh.faces[iFaceIndex].patchIndex
                boundaryType = velocity[iBoundary].type
                if boundaryType == "fixedValue"
                    fluxCb = nu * theFace.gDiff
                    relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                    fluxVb = velocity[iBoundary].values[relativeFaceIndex] .* -fluxCb
                    RHS[iElement,:] .-= fluxVb
                    diag += fluxCb
                end
            end
        end
        push!(rows, iElement)
        push!(cols, iElement)
        push!(vals, MVector(diag, diag, diag))
    end

    return SparseArrays.sparse(
        rows,
        cols,
        vals,
    ), RHS
end # function cellBasedAssemblySparseMultiVectorPrealloc

function LdcCellBasedAssemblySparseMatrix(input::LdcMatrixAssemblyInput)::Tuple{SparseMatrixCSC{Float64,Int64},SparseMatrixCSC{Float64,Int64}}
    # function LdcCellBasedAssemblySparseMatrix(input::LdcMatrixAssemblyInput)::Tuple{SparseMatrixCSC{MVector{3,Float64},Int64},SparseMatrixCSC{Float64,Int64}}
    mesh = input.mesh
    nu = input.nu
    # pressure = input.p  ignore for now
    velocity = input.U
    nCells = size(mesh.cells)[1]
    coeffMatrices = spzeros(nCells, nCells) # ux, uy, uz (ignore pressure for now)
    # coeffMatrices = spzeros(MVector{3,Float64}, nCells, nCells) # ux, uy, uz (ignore pressure for now)
    # coeffMatrices = [spzeros(nCells, nCells) for _ in 1:3] 
    RHS = spzeros(nCells)  # [ux,uy,uz]
    # RHS = spzeros(nCells, 3)  # [ux,uy,uz]
    progress = ProgressBar(total=nCells)
    for (iElement, theElement) in enumerate(mesh.cells[1:50000])
        update(progress)
        # assume no body forces
        # for iVar in 1:numVariables
        #     RHS[iVar, iElement] *= theElement.volume  # S_u
        # end
        for (iFace, iFaceIndex) in enumerate(theElement.iFaces)
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor != -1
                fluxCn = nu * theFace.gDiff
                fluxFn = -fluxCn
                # U
                coeffMatrices[iElement, theElement.iNeighbors[iFace]] = fluxFn
                # coeffMatrices[iElement, theElement.iNeighbors[iFace]] = @MVector fill(fluxFn, 3)
                coeffMatrices[iElement, iElement] += fluxCn

                # ignore p for now
                # coeffMatrices[2][iElement, theElement.iNeighbors[iFace]] = 0.0
                # coeffMatrices[2][iElement, iElement] += 0.0
            else
                iBoundary = mesh.faces[iFaceIndex].patchIndex
                boundaryType = velocity[iBoundary].type
                if boundaryType == "fixedValue"
                    fluxCb = nu * theFace.gDiff
                    relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace

                    fluxVb = velocity[iBoundary].values[relativeFaceIndex] .* -fluxCb
                    coeffMatrices[iElement, iElement] += fluxCb
                    # coeffMatrices[iElement, iElement] = coeffMatrices[iElement, iElement].+ fluxCb
                    RHS[iElement] -= fluxVb[1]
                    # for dim in 1:3
                    #     fluxVb = -fluxCb * velocity[iBoundary].values[relativeFaceIndex][dim] 
                    #     coeffMatrices[dim][iElement, iElement] += fluxCb
                    #     RHS[iElement, dim] -= fluxVb
                    # end
                end
                # TODO ignore pressure for now ig
                # boundaryType = pressure[iBoundary].type
                # if boundaryType == "fixedValue"
                #     fluxCb = 0 # FIXME ?
                #     relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                #     fluxVb = -fluxCb * pressure[iBoundary].values[relativeFaceIndex]
                #     coeffMatrices[2][iElement, iElement] += fluxCb
                # end
            end
        end
    end
    return coeffMatrices, RHS
end # function genericCellBasedAssemblySparseMatrix


export cellBasedAssembly,
    cellBasedAssemblySparse1,
    cellBasedAssemblySparse2,
    cellBasedAssemblySparse3,
    faceBasedAssembly,
    batchedFaceBasedAssembly,
    batchedFaceBasedAssemblySparseMatrix,
    LdcCellBasedAssemblySparseMultiVectorPrealloc
end # module MatrixAssembly
