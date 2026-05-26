using CUDA
using SplitApplyCombine
include("../classes.jl")
function getCellBasedGpuInput(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    nus = CuArray(input.nu)
    gpuFaces = toGPUSOAs(VectorToSOAs(input.mesh.faces))
    iFaces, iNeighbors, numInts, iFaceOffsets, numFaces = flattenCells(input)
    bFaceValues, U, bFaceMapping = gpu_getFaceValues(input)
    nCells::Int32 = length(input.mesh.cells)
    RHS = CUDA.zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(input.mesh.cells) + 2 * input.mesh.numInteriorFaces
    vals = CUDA.zeros(P, entriesNeeded)
    rowoffsets = input.offsets |> cu 
    return iFaces, iNeighbors, numInts, iFaceOffsets, numFaces, nus, gpuFaces.Sf, gpuFaces.gDiff, U, rowoffsets, gpuFaces.ownerRelOwnerIdx, gpuFaces.neighborRelNeighborIdx, bFaceValues, bFaceMapping, gpuFaces.iOwner, vals, RHS
end

function faceInput(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    gpuFaces = toGPUSOAs(VectorToSOAs(input.mesh.faces))
    bFaceValues, U, bFaceMapping = gpu_getFaceValues(input)
    nus = CuArray(input.nu)
    entriesNeeded::Int32 = length(input.mesh.cells) + 2 * input.mesh.numInteriorFaces
    RHS = CUDA.zeros(P, length(input.mesh.cells) * 3)
    vals = CUDA.zeros(P, entriesNeeded)
    return gpuFaces, U, nus, bFaceValues, bFaceMapping, vals, RHS
end

function getGreedyEdgeColoring(input::MatrixAssemblyInput)
    mesh = input.mesh
    for face::Face in mesh.faces
        face.batchId = -1
    end
    faceColorMapping = zeros(Int32, mesh.numInteriorFaces)
    for cell::Cell in mesh.cells
        usedColors = []
        for iFace in 1:cell.nInternalFaces
            iFaceIndex = cell.iFaces[iFace]
            face::Face = mesh.faces[iFaceIndex]
            if face.batchId == -1
                continue
            end
            push!(usedColors, face.batchId)
        end
        id = 1
        for iFace in 1:cell.nInternalFaces
            iFaceIndex = cell.iFaces[iFace]
            face::Face = mesh.faces[iFaceIndex]
            if face.batchId != -1
                continue
            end
            go = true
            while go
                if id in usedColors
                    id += 1
                    continue
                end
                face.batchId = id
                faceColorMapping[face.index] = id
                push!(usedColors, face.batchId)
                go = false
            end
        end
    end
    return maximum(faceColorMapping), faceColorMapping
end


function getBatchedFaceBasedGpuInput(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    colors, _ = getGreedyEdgeColoring(input)
    grouped = group(x -> x.batchId, input.mesh.faces)
    gpubatches = [toGPUSOAs(VectorToSOAs(bx)) for bx in grouped.values]
    bFaceValues, U, bFaceMapping = gpu_getFaceValues(input)
    nus = CuArray(input.nu)
    entriesNeeded::Int32 = length(input.mesh.cells) + 2 * input.mesh.numInteriorFaces
    RHS = CUDA.zeros(P, length(input.mesh.cells) * 3)
    vals = CUDA.zeros(P, entriesNeeded)
    return gpubatches, U, nus, bFaceValues, bFaceMapping, vals, RHS, colors
end

function flattenCells(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    nIFaces = [length(c.iFaces) for c in input.mesh.cells]
    iFaces = fill(-1, sum(nIFaces))
    iNeighbors = fill(-1, sum(nIFaces))
    iFaceOffsets = ones(Int32, length(input.mesh.cells))
    iFaces[1:1+length(input.mesh.cells[1].iFaces)-1] = input.mesh.cells[1].iFaces
    iNeighbors[1:1+length(input.mesh.cells[1].iNeighbors)-1] = input.mesh.cells[1].iNeighbors
    for iCell in 2:length(input.mesh.cells)
        cell = input.mesh.cells[iCell]
        start = Int32(iFaceOffsets[iCell-1] + length(input.mesh.cells[iCell-1].iFaces))
        iFaceOffsets[iCell] = start
        iFaces[start:start+length(cell.iFaces)-1] = cell.iFaces
        iNeighbors[start:start+length(cell.iNeighbors)-1] = cell.iNeighbors
    end
    numInts = [c.nInternalFaces for c in input.mesh.cells]
    return CuArray(iFaces), CuArray(iNeighbors), CuArray(numInts), CuArray(iFaceOffsets), CuArray(nIFaces)
end