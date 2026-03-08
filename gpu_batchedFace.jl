include("init.jl")
include("gpu_faceBased.jl")
include("gpu_helper.jl")
function gpu_LaplaceOnlyBatchedFaceAssemblyRunner(iOwners::CuArray{Int32},
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
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,Float32}},
    Sf::CuArray{SVector{3,Float32}},
    bFaceMapping::CuArray{Int32}
)
    for color in 1:numBatches
        iblocks = cld(N, 256)
        CUDA.@sync @cuda threads = 256 blocks = iblocks kernel_LaplaceOnlyBatchedFaceAssembly(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, faceColorMapping, color)
    end
    bblocks = cld(M, 256)

    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace_LaplaceOnly(iOwners, gDiffs, offsets, nu_g, vals, entriesNeeded, bFaceValues, RHS, N, nCells, M, bFaceMapping)
end

function kernel_LaplaceOnlyBatchedFaceAssembly(
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
    color,
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

    # Diffusion
    diffusion = nus[iOwner] * gDiffs[iFace]

    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    CUDA.@atomic vals[idx] += diffusion    # x
    CUDA.@atomic vals[idx+entriesNeeded] += diffusion  # y
    CUDA.@atomic vals[idx+entriesNeeded+entriesNeeded] += diffusion  # z

    idx = offsets[iOwner] + relativeToOwner[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] -= diffusion    # x
    vals[idx+entriesNeeded] -= diffusion  # y
    vals[idx+entriesNeeded+entriesNeeded] -= diffusion  # z

    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    CUDA.@atomic vals[idx] -= diffusion    # x
    CUDA.@atomic vals[idx+entriesNeeded] -= diffusion  # y
    CUDA.@atomic vals[idx+entriesNeeded+entriesNeeded] -= diffusion  # z

    idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += diffusion    # x
    vals[idx+entriesNeeded] += diffusion  # y
    vals[idx+entriesNeeded+entriesNeeded] += diffusion  # z
    return nothing
end

function gpu_DivOnlyPrecalculatedWeightsBatchedFaceAssemblyRunner(
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
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,Float32}},
    Sf::CuArray{SVector{3,Float32}},
    bFaceMapping::CuArray{Int32},
    weights::CuArray{Float32}
)
    for color in 1:numBatches
        iblocks = cld(N, 256)
        CUDA.@sync @cuda threads = 256 blocks = iblocks kernel_DivOnlyPrecalculatedWeightsBatchedFaceAssembly(iOwners, iNeighbors, offsets, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, faceColorMapping, color, U, Sf, weights)
    end
    bblocks = cld(M, 256)

    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace_DivOnly(iOwners, bFaceValues, RHS, N, nCells, M, Sf, bFaceMapping)
end

function kernel_DivOnlyPrecalculatedWeightsBatchedFaceAssembly(
    iOwners,
    iNeighbors,
    offsets,
    rows,
    cols,
    vals,
    entriesNeeded,
    relativeToOwner,
    numInteriorFaces,
    relativeToNeighbor,
    faceColors,
    color,
    U,
    Sf,
    weights
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

    # Convection
    Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
    ϕf::Float32 = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = weights[iFace]                              # get weight of transport variable interpolation 
    # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
    valueUpper::Float32 = ϕf * weights_f
    valueLower::Float32 = -ϕf * (1 - weights_f)

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

function gpu_DivOnlyHardcodedDivBatchedFaceAssemblyRunner(
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
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,Float32}},
    Sf::CuArray{SVector{3,Float32}},
    bFaceMapping::CuArray{Int32},
    div::String
)
    if div == "CDF"
        for color in 1:numBatches
            iblocks = cld(N, 256)
            CUDA.@sync @cuda threads = 256 blocks = iblocks kernel_DivOnlyHardcodedDivCDFBatchedFaceAssembly(iOwners, iNeighbors, offsets, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, faceColorMapping, color, U, Sf)
        end
    else
        for color in 1:numBatches
            iblocks = cld(N, 256)
            CUDA.@sync @cuda threads = 256 blocks = iblocks kernel_DivOnlyHardcodedDivUpwindBatchedFaceAssembly(iOwners, iNeighbors, offsets, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, faceColorMapping, color, U, Sf)
        end
    end
    bblocks = cld(M, 256)

    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace_DivOnly(iOwners, bFaceValues, RHS, N, nCells, M, Sf, bFaceMapping)
end


function kernel_DivOnlyHardcodedDivCDFBatchedFaceAssembly(
    iOwners,
    iNeighbors,
    offsets,
    rows,
    cols,
    vals,
    entriesNeeded,
    relativeToOwner,
    numInteriorFaces,
    relativeToNeighbor,
    faceColors,
    color,
    U,
    Sf,
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

    # Convection
    Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
    ϕf::Float32 = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = centralDifferencing(ϕf)                             # get weight of transport variable interpolation 
    # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
    valueUpper::Float32 = ϕf * weights_f
    valueLower::Float32 = -ϕf * (1 - weights_f)

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



function kernel_DivOnlyHardcodedDivUpwindBatchedFaceAssembly(
    iOwners,
    iNeighbors,
    offsets,
    rows,
    cols,
    vals,
    entriesNeeded,
    relativeToOwner,
    numInteriorFaces,
    relativeToNeighbor,
    faceColors,
    color,
    U,
    Sf,
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

    # Convection
    Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
    ϕf::Float32 = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = upwind(ϕf)                             # get weight of transport variable interpolation 
    # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
    valueUpper::Float32 = ϕf * weights_f
    valueLower::Float32 = -ϕf * (1 - weights_f)

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


function gpu_DivOnlyDynamicBatchedFaceAssemblyRunner(
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
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,Float32}},
    Sf::CuArray{SVector{3,Float32}},
    bFaceMapping::CuArray{Int32},
    div::Function
)
    for color in 1:numBatches
        iblocks = cld(N, 256)
        CUDA.@sync @cuda threads = 256 blocks = iblocks kernel_DivOnlyDynamicBatchedFaceAssembly(iOwners, iNeighbors, offsets, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, faceColorMapping, color, U, Sf, div)
    end
    bblocks = cld(M, 256)

    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace_DivOnly(iOwners, bFaceValues, RHS, N, nCells, M, Sf, bFaceMapping)
end

function kernel_DivOnlyDynamicBatchedFaceAssembly(
    iOwners,
    iNeighbors,
    offsets,
    rows,
    cols,
    vals,
    entriesNeeded,
    relativeToOwner,
    numInteriorFaces,
    relativeToNeighbor,
    faceColors,
    color,
    U,
    Sf,
    div
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

    # Convection
    Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
    ϕf::Float32 = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = div(ϕf)                             # get weight of transport variable interpolation 
    # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
    valueUpper::Float32 = ϕf * weights_f
    valueLower::Float32 = -ϕf * (1 - weights_f)

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

function gpu_PrecalculatedWeightsBatchedFaceAssemblyRunner(
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
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,Float32}},
    Sf::CuArray{SVector{3,Float32}},
    bFaceMapping::CuArray{Int32},
    weights::CuArray{Float32}
)
    for color in 1:numBatches
        iblocks = cld(N, 256)
        CUDA.@sync @cuda threads = 256 blocks = iblocks kernel_PrecalculatedWeightsBatchedFaceAssembly(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, faceColorMapping, color, U, Sf, weights)
    end
    bblocks = cld(M, 256)

    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace(iOwners, gDiffs, offsets, nu_g, vals, entriesNeeded, bFaceValues, RHS, N, nCells, M, Sf, bFaceMapping)
end

function kernel_PrecalculatedWeightsBatchedFaceAssembly(
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
    color,
    U,
    Sf,
    weights
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

    # Diffusion
    diffusion = nus[iOwner] * gDiffs[iFace]

    # Convection
    Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
    ϕf::Float32 = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = weights[iFace]                              # get weight of transport variable interpolation 
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


function gpu_HardcodedBatchedFaceAssemblyRunner(iOwners::CuArray{Int32},
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
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,Float32}},
    Sf::CuArray{SVector{3,Float32}},
    bFaceMapping::CuArray{Int32},
    div::String
)
    if div == "CDF"
        for color in 1:numBatches
            iblocks = cld(N, 256)
            CUDA.@sync @cuda threads = 256 blocks = iblocks kernel_HardcodedCDFBatchedFaceAssembly(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, faceColorMapping, color, U, Sf)
        end
    else
        for color in 1:numBatches
            iblocks = cld(N, 256)
            CUDA.@sync @cuda threads = 256 blocks = iblocks kernel_HardcodedUpwindBatchedFaceAssembly(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, faceColorMapping, color, U, Sf)
        end
    end
    bblocks = cld(M, 256)

    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace(iOwners, gDiffs, offsets, nu_g, vals, entriesNeeded, bFaceValues, RHS, N, nCells, M, Sf, bFaceMapping)
end

function kernel_HardcodedCDFBatchedFaceAssembly(
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
    color,
    U,
    Sf,
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

    # Diffusion
    diffusion = nus[iOwner] * gDiffs[iFace]

    # Convection
    Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
    ϕf::Float32 = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = centralDifferencing(ϕf)                              # get weight of transport variable interpolation 
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


function kernel_HardcodedUpwindBatchedFaceAssembly(
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
    color,
    U,
    Sf,
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

function gpu_DynamicBatchedFaceAssemblyRunner(
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
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,Float32}},
    Sf::CuArray{SVector{3,Float32}},
    bFaceMapping::CuArray{Int32},
    div::Function
)
    for color in 1:numBatches
        iblocks = cld(N, 256)
        CUDA.@sync @cuda threads = 256 blocks = iblocks kernel_DynamicBatchedFaceAssembly(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, faceColorMapping, color, U, Sf, div)
    end
    bblocks = cld(M, 256)

    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace(iOwners, gDiffs, offsets, nu_g, vals, entriesNeeded, bFaceValues, RHS, N, nCells, M, Sf, bFaceMapping)
end

function kernel_DynamicBatchedFaceAssembly(
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
    color,
    U,
    Sf,
    div
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

    # Diffusion
    diffusion = nus[iOwner] * gDiffs[iFace]

    # Convection
    Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
    ϕf::Float32 = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = div(ϕf)                              # get weight of transport variable interpolation 
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




function gpu_BatchedFaceBasedAssembly(input::MatrixAssemblyInput)
    iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, numBlocks, bFaceValues, RHS, nCells, M, numBatches, faceColorMapping, U, Sf, bFaceMapping = gpu_prepareBatchedFaceBased(input)
    batchedFaceBasedRunner(
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
        N,
        relativeToNbs,
        numBlocks,
        bFaceValues,
        RHS,
        nCells,
        M,
        numBatches,
        faceColorMapping,
        U,
        Sf,
        bFaceMapping
    )
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
    numBlocks::Int32,
    bFaceValues::CuArray{SVector{3,Float32}},
    RHS::CuArray{Float32},
    nCells::Int32,
    M::Int32,
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,Float32}},
    Sf::CuArray{SVector{3,Float32}},
    bFaceMapping::CuArray{Int32},
)
    for color in 1:numBatches
        iblocks = cld(N, 256)
        CUDA.@sync @cuda threads = 256 blocks = iblocks kernel_internalFace_coloured(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, faceColorMapping, color, U, Sf)
    end
    bblocks = cld(M, 256)

    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace(iOwners, gDiffs, offsets, nu_g, vals, entriesNeeded, bFaceValues, RHS, N, nCells, M, Sf, bFaceMapping)
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
    color,
    U,
    Sf
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
    return nothing
end

############################# helper 


function gpu_prepareBatchedFaceBased(input::MatrixAssemblyInput)
    iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, numBlocks, bFaceValues, RHS, nCells, M, U, Sf, bFaceMapping = gpu_prepareFaceBased(input)
    numBatches, faceColorMapping = getGreedyEdgeColoring(input)
    return iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, numBlocks, bFaceValues, RHS, nCells, M, numBatches, faceColorMapping, U, Sf, bFaceMapping
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
            face::Face = mesh.faces[iFace]
            if face.batchId == -1
                continue
            end
            push!(usedColors, face.batchId)
        end
        id = 1
        for iFace in 1:cell.nInternalFaces
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



###### provable not really useful




function batchedFaceBasedAllRunner(
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
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,Float32}},
    Sf::CuArray{SVector{3,Float32}},
    bFaceMapping::CuArray{Int32}
)
    for color in 1:numBatches
        CUDA.@sync @cuda threads = 256 blocks = numBlocks kernel_all_colored(
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
            faceColorMapping,
            color,
            bFaceMapping
        )
    end
    return rows, cols, vals, RHS
end
function kernel_all_colored(
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
    faceColors,
    color,
    bFaceMapping
)
    iFace = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if iFace > numInternalFaces + numBoundaryFaces
        return
    end
    iOwner = iOwners[iFace]
    if faceColors[iFace] != color
        return
    end
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
        if bFaceIndex == -1 # boundarytype != fixed
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