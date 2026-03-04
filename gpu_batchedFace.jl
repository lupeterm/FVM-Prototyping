include("init.jl")
include("gpu_facebased.jl")

function gpu_BatchedFaceBasedAssembly(input::MatrixAssemblyInput)
    iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, internal_blocks, bblocks, bFaceValues, RHS, nCells, M, numBatches, faceColorMapping = gpu_prepareBatchedFaceBased(input)
    batchedFaceBasedRunner(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, internal_blocks, bblocks, bFaceValues, RHS, nCells, M, numBatches, faceColorMapping)
    # return Vector(rows), Vector(cols), Vector(vals), Vector(RHS)
    return rows, cols, vals, RHS
end

function batchedFaceBasedRunner(
    iOwners::CuArray{Int32},
    iNeighbors::CuArray{Int32},
    gDiffs::CuArray{Float32},
    offsets::CuArray{Int32},
    nu_g::CuArray{Float32},
    rows::CuArray{Int32},
    cols::CuArray{Int32},
    vals::CuArray{Float32},
    entriesNeeded::Int32,
    relativeToOwners::CuArray{Int32},
    N::Int32,
    relativeToNbs::CuArray{Int32},
    internalblocks::Int32,
    bblocks::Int32,
    bFaceValues::CuArray{Float32},
    RHS::CuArray{Float32},
    nCells::Int32,
    M::Int32,
    numBatches::Int32,
    faceColorMapping::CuArray{Int32}
)
    for color in 1:numBatches
        CUDA.@sync @cuda threads = 256 blocks = internalblocks kernel_internalFace_coloured(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, faceColorMapping, color)
    end
    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace(iOwners, gDiffs, offsets, nu_g, vals, entriesNeeded, bFaceValues, RHS, N, nCells, M)
end

function kernel_internalFace_coloured(
    iOwners,
    iNeighbors,
    gDiffs,
    offsets,
    nus,
    rows,
    cols,
    vals,
    entriesNeeded,
    relativeToOwner,
    numInteriorFaces,
    relativeToNeighbor,
    faceColors,
    color
)
    iFace = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if iFace > numInteriorFaces
        return
    end
    faceColor = faceColors[iFace]
    if faceColor != color
        return
    end
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    fluxCn = nus[iOwner] * gDiffs[iFace]
    fluxFn = -fluxCn
    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    vals[idx] += fluxCn    # x
    vals[idx+entriesNeeded] += fluxCn  # y
    vals[idx+entriesNeeded+entriesNeeded] += fluxCn  # z

    idx = offsets[iOwner] + relativeToOwner[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] += fluxFn    # x
    vals[idx+entriesNeeded] += fluxFn  # y
    vals[idx+entriesNeeded+entriesNeeded] += fluxFn  # z

    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    vals[idx] += fluxCn    # x
    vals[idx+entriesNeeded] += fluxCn  # y
    vals[idx+entriesNeeded+entriesNeeded] += fluxCn  # z

    idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += fluxFn    # x
    vals[idx+entriesNeeded] += fluxFn  # y
    vals[idx+entriesNeeded+entriesNeeded] += fluxFn  # z
    return nothing
end

############################# helper 


function gpu_prepareBatchedFaceBased(input::MatrixAssemblyInput)
    iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, internal_blocks, bblocks, bFaceValues, RHS, nCells, M = gpu_prepareFaceBased(input)
    numBatches, faceColorMapping = getGreedyEdgeColoring(input)
    return iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, internal_blocks, bblocks, bFaceValues, RHS, nCells, M, numBatches, faceColorMapping
end


function getGreedyEdgeColoring(input::MatrixAssemblyInput)
    mesh = input.mesh
    for face::Face in mesh.faces
        face.batchId = -1
    end
    faceColorMapping = zeros(Int32, mesh.numInteriorFaces)
    for cell::Cell in mesh.cells
        usedColors = []
        for iFace in cell.iFaces[1:cell.nInternalFaces]
            face::Face = mesh.faces[iFace]
            if face.batchId == -1
                continue
            end
            push!(usedColors, face.batchId)
        end
        id = 1
        for iFace in cell.iFaces[1:cell.nInternalFaces]
            face::Face = mesh.faces[iFace]
            if face.batchId != -1
                continue
            end
            while true
                if id in usedColors
                    id += 1
                    continue
                end
                face.batchId = id
                faceColorMapping[face.index] = id
                push!(usedColors, face.batchId)
                break
            end
        end
    end
    return maximum(faceColorMapping), CuArray{Int32}(faceColorMapping)
end



