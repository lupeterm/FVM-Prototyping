include("init.jl")
include("gpu_helper.jl")

function gpu_FaceBasedAssembly(input::MatrixAssemblyInput)
    iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, internal_blocks, bblocks, bFaceValues, RHS, nCells, M = gpu_prepareFaceBased(input)
    faceBasedRunner(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, internal_blocks, bblocks, bFaceValues, RHS, nCells, M)
    return Vector(rows), Vector(cols), Vector(vals), Vector(RHS)
end

function gpu_DivOnlyBatchedFaceAssembly(input::MatrixAssemblyInput)
end
function gpu_LaplaceOnlyBatchedFaceAssembly(input::MatrixAssemblyInput)
end
function gpu_PrecalculatedWeightsUpwindBatchedFaceAssembly(input::MatrixAssemblyInput)
end
function gpu_PrecalculatedWeightsCDFBatchedFaceAssembly(input::MatrixAssemblyInput)
end
function gpu_HardCodedUpwindBatchedFaceAssembly(input::MatrixAssemblyInput)
end
function gpu_HardCodedCDFBatchedFaceAssembly(input::MatrixAssemblyInput)
end
function gpu_DynamicCDFBatchedFaceAssembly(input::MatrixAssemblyInput)
end
function gpu_DynamicUpwindBatchedFaceAssembly(input::MatrixAssemblyInput)
end

function faceBasedRunner(
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
    M::Int32
)
    CUDA.@sync @cuda threads = 256 blocks = internalblocks kernel_internalFace(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs)
    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace(iOwners, gDiffs, offsets, nu_g, vals, entriesNeeded, bFaceValues, RHS, N, nCells, M)
end

function kernel_internalFace(
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
    relativeToNeighbor
)
    iFace = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if iFace > numInteriorFaces
        return
    end
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    fluxCn = nus[iOwner] * gDiffs[iFace]
    fluxFn = -fluxCn

    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    CUDA.@atomic vals[idx] += fluxCn    # x
    CUDA.@atomic vals[idx+entriesNeeded] += fluxCn  # y
    CUDA.@atomic vals[idx+entriesNeeded+entriesNeeded] += fluxCn  # z

    idx = offsets[iOwner] + relativeToOwner[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] += fluxFn    # x
    vals[idx+entriesNeeded] += fluxFn  # y
    vals[idx+entriesNeeded+entriesNeeded] += fluxFn  # z

    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    CUDA.@atomic vals[idx] += fluxCn    # x
    CUDA.@atomic vals[idx+entriesNeeded] += fluxCn  # y
    CUDA.@atomic vals[idx+entriesNeeded+entriesNeeded] += fluxCn  # z

    idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += fluxFn    # x
    vals[idx+entriesNeeded] += fluxFn  # y
    vals[idx+entriesNeeded+entriesNeeded] += fluxFn  # z
    return nothing
end


function kernel_boundaryFace(
    iOwners,
    gDiffs,
    offsets,
    nus,
    vals,
    entriesNeeded,
    bFaceValues,
    RHS,
    numInternalFaces,
    nCells,
    numBoundaryFaces
)
    localBFaceIndex = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    globalFaceIndex = numInternalFaces + localBFaceIndex
    if globalFaceIndex > numInternalFaces + numBoundaryFaces
        return
    end
    if globalFaceIndex <= numInternalFaces
        return
    end
    iOwner = iOwners[globalFaceIndex]
    fluxCb = nus[iOwner] * gDiffs[globalFaceIndex]
    boundaryValueIndex = (localBFaceIndex - 1) * 3 + 1  # one based indexing :grr:
    x = bFaceValues[boundaryValueIndex]
    y = bFaceValues[boundaryValueIndex+1]
    z = bFaceValues[boundaryValueIndex+2]
    fluxVbx = x * -fluxCb
    fluxVby = y * -fluxCb
    fluxVbz = z * -fluxCb
    idx = offsets[iOwner]
    CUDA.@atomic vals[idx] += fluxCb    # x
    CUDA.@atomic vals[idx+entriesNeeded] += fluxCb  # y
    CUDA.@atomic vals[idx+entriesNeeded+entriesNeeded] += fluxCb  # z
    # if fluxVbz != 0.0
    #     @cuprintln("fluxVbz != 0.0: $fluxVbz, gdiff: $(gDiffs[globalFaceIndex]), [$x, $y, $z], [$(bFaceValues[1]), $(bFaceValues[2]), $(bFaceValues[3])]")
    # end
    CUDA.@atomic RHS[iOwner] -= fluxVbx
    CUDA.@atomic RHS[iOwner+nCells] -= fluxVby
    CUDA.@atomic RHS[iOwner+nCells+nCells] -= fluxVbz
    return nothing
end