include("init.jl")
include("gpu_helper.jl")

function gpu_FaceBasedAssembly(input::MatrixAssemblyInput)
    iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, numBlocks, bFaceValues, RHS, nCells, M, U, Sf,bFaceMapping = gpu_prepareFaceBased(input)
    faceBasedAllRunner(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, numBlocks, bFaceValues, RHS, nCells, M, U, Sf, bFaceMapping)
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


function faceBasedAllRunner(
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
    numBlocks::Int32,
    bFaceValues::CuArray{SVector{3,Float32}},
    RHS::CuArray{Float32},
    nCells::Int32,
    M::Int32,
    U::CuArray{SVector{3,Float32}},
    Sf::CuArray{SVector{3,Float32}},
    bFaceMapping::CuArray{Int32}
)
    # blocks = cld(N + M, 256)
    # println(blocks)
    CUDA.@sync @cuda threads = 256 blocks = cld(length(iOwners), 256) kernel_all(
        iOwners,
        iNeighbors,
        gDiffs,
        offsets,
        nu_g,
        rows,
        cols,
        vals,
        entriesNeeded,
        relativeToOwners,
        relativeToNbs,
        U,
        Sf,
        bFaceValues,
        RHS,
        N,
        nCells,
        M,
        bFaceMapping
    )
end

function kernel_all(
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
    relativeToNeighbor,
    U,
    Sf,
    bFaceValues,
    RHS,
    numInternalFaces,
    nCells,
    numBoundaryFaces,
    bFaceMapping
)
    iFace = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if iFace > numInternalFaces + numBoundaryFaces
        return
    end
    
    iOwner = iOwners[iFace]
    if iFace <= numInternalFaces

        iNeighbor = iNeighbors[iFace]
        # Diffusion
        diffusion = nus[iOwner] * gDiffs[iFace]

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf::Float32 = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = 0.5                              # get weight of transport variable interpolation 
                                                            # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::Float32 = ϕf * weights_f + diffusion
        valueLower::Float32 = -ϕf * (1 - weights_f) - diffusion

        idx = offsets[iOwner]
        cols[idx] = iOwner
        rows[idx] = iOwner
        CUDA.@atomic vals[idx] += valueUpper    # x
        CUDA.@atomic vals[idx+entriesNeeded] += valueUpper  # y
        CUDA.@atomic vals[idx+entriesNeeded+entriesNeeded] += valueUpper  # z

        idx = offsets[iOwner] + relativeToOwner[iFace]
        cols[idx] = iOwner
        rows[idx] = iNeighbor
        vals[idx] += valueLower    # x
        vals[idx+entriesNeeded] += valueLower  # y
        vals[idx+entriesNeeded+entriesNeeded] += valueLower  # z

        idx = offsets[iNeighbor]
        cols[idx] = iNeighbor
        rows[idx] = iNeighbor
        CUDA.@atomic vals[idx] += valueLower    # x
        CUDA.@atomic vals[idx+entriesNeeded] += valueLower  # y
        CUDA.@atomic vals[idx+entriesNeeded+entriesNeeded] += valueLower  # z

        idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
        cols[idx] = iNeighbor
        rows[idx] = iOwner
        vals[idx] += valueUpper    # x
        vals[idx+entriesNeeded] += valueUpper  # y
        vals[idx+entriesNeeded+entriesNeeded] += valueUpper  # z
    else
        relativeFaceIndex = iFace - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex == -1
            return
        end
        convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])
        diffusion = bFaceValues[bFaceIndex] .* nus[iOwner] * gDiffs[iFace]
        idx = offsets[iOwner]
        CUDA.@atomic vals[idx] -= diffusion[1]    # x
        CUDA.@atomic vals[idx+entriesNeeded] -= diffusion[2]  # y
        CUDA.@atomic vals[idx+entriesNeeded+entriesNeeded] -= diffusion[3]  # z

        # RHS/Source
        value = convection .+ diffusion
        CUDA.@atomic RHS[iOwner] -= value[1]
        CUDA.@atomic RHS[iOwner+nCells] -= value[2]
        CUDA.@atomic RHS[iOwner+nCells+nCells] -= value[3]
    end
    return nothing
end

##### Proven to be inferior to joined kernel with branches

function SplitfaceBasedRunner(
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
    bblocks::Int32,
    bFaceValues::CuArray{SVector{3,Float32}},
    RHS::CuArray{Float32},
    nCells::Int32,
    M::Int32,
    U::CuArray{SVector{3,Float32}},
    Sf::CuArray{SVector{3,Float32}},
    bFaceMapping::CuArray{Int32}
)
    iblocks = cld(N, 256)
    bblocks = cld(M, 256)

    CUDA.@sync @cuda threads = 256 blocks = iblocks kernel_internalFace(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, U, Sf)
    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace(iOwners, gDiffs, offsets, nu_g, vals, entriesNeeded, bFaceValues, RHS, N, nCells, M, Sf, bFaceMapping)
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
    relativeToNeighbor,
    U,
    Sf,
)
    iFace = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if iFace > numInteriorFaces
        return
    end
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]

    # Diffusion
    diffusion = nus[iOwner] * gDiffs[iFace]

    # Convection
    Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
    ϕf::Float32 = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = upwind(ϕf)                              # get weight of transport variable interpolation 
    # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
    valueUpper::Float32 = ϕf * weights_f + diffusion
    valueLower::Float32 = -ϕf * (1 - weights_f) - diffusion

    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    CUDA.@atomic vals[idx] += valueUpper    # x
    CUDA.@atomic vals[idx+entriesNeeded] += valueUpper  # y
    CUDA.@atomic vals[idx+entriesNeeded+entriesNeeded] += valueUpper  # z

    idx = offsets[iOwner] + relativeToOwner[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x
    vals[idx+entriesNeeded] += valueLower  # y
    vals[idx+entriesNeeded+entriesNeeded] += valueLower  # z

    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    CUDA.@atomic vals[idx] += valueLower    # x
    CUDA.@atomic vals[idx+entriesNeeded] += valueLower  # y
    CUDA.@atomic vals[idx+entriesNeeded+entriesNeeded] += valueLower  # z

    idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x
    vals[idx+entriesNeeded] += valueUpper  # y
    vals[idx+entriesNeeded+entriesNeeded] += valueUpper  # z
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
    numBoundaryFaces,
    Sf,
    bFaceMapping
)
    iFace = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    globalFaceIndex = numInternalFaces + iFace
    if globalFaceIndex > numInternalFaces + numBoundaryFaces
        return
    end
    if globalFaceIndex <= numInternalFaces
        return
    end
    bFaceIndex = bFaceMapping[iFace]
    if bFaceIndex == -1
        return
    end
    iOwner = iOwners[globalFaceIndex]
    convection = bFaceValues[bFaceIndex] .* dot(Sf[globalFaceIndex], bFaceValues[bFaceIndex])
    diffusion = bFaceValues[bFaceIndex] .* nus[iOwner] * gDiffs[globalFaceIndex]
    idx = offsets[iOwner]
    CUDA.@atomic vals[idx] -= diffusion[1]    # x
    CUDA.@atomic vals[idx+entriesNeeded] -= diffusion[2]  # y
    CUDA.@atomic vals[idx+entriesNeeded+entriesNeeded] -= diffusion[3]  # z

    # RHS/Source
    value = convection .+ diffusion
    CUDA.@atomic RHS[iOwner] -= value[1]
    CUDA.@atomic RHS[iOwner+nCells] -= value[2]
    CUDA.@atomic RHS[iOwner+nCells+nCells] -= value[3]
    return nothing
end
