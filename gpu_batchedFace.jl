include("init.jl")
include("gpu_faceBased.jl")
include("gpu_helper.jl")

function gpu_LaplaceOnlyBatchedFaceAssemblyRunner(
    iOwners::CuArray{Int32},
    iNeighbors::CuArray{Int32},
    gDiffs::CuArray{P},
    offsets::CuArray{Int32},
    nu_g::CuArray{P},
    rows::CuArray{Int32},
    cols::CuArray{Int32},
    vals::CuArray{P},
    entriesNeeded::Int32,
    relativeToOwners::CuArray{Int32},
    N::Int32,
    relativeToNbs::CuArray{Int32},
    numBlocks::Int32,
    bFaceValues::CuArray{SVector{3,P}},
    RHS::CuArray{P},
    nCells::Int32,
    M::Int32,
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32}
) where {P<:AbstractFloat}
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
    vals[idx] += diffusion    # x

    idx = offsets[iOwner] + relativeToOwner[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] -= diffusion    # x

    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    vals[idx] -= diffusion    # x

    idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += diffusion    # x
    return nothing
end

function gpu_DivOnlyPrecalculatedWeightsBatchedFaceAssemblyRunner(
    iOwners::CuArray{Int32},
    iNeighbors::CuArray{Int32},
    gDiffs::CuArray{P},
    offsets::CuArray{Int32},
    nu_g::CuArray{P},
    rows::CuArray{Int32},
    cols::CuArray{Int32},
    vals::CuArray{P},
    entriesNeeded::Int32,
    relativeToOwners::CuArray{Int32},
    N::Int32,
    relativeToNbs::CuArray{Int32},
    numBlocks::Int32,
    bFaceValues::CuArray{SVector{3,P}},
    RHS::CuArray{P},
    nCells::Int32,
    M::Int32,
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32},
    weights::CuArray{P}
) where {P<:AbstractFloat}
    for color in 1:numBatches
        iblocks = cld(N, 256)
        CUDA.@sync @cuda threads = 256 blocks = iblocks kernel_DivOnlyPrecalculatedWeightsBatchedFaceAssembly(iOwners, iNeighbors, offsets, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, faceColorMapping, color, U, Sf, weights)
    end
    bblocks = cld(M, 256)

    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace_DivOnly(iOwners, bFaceValues, RHS, N, nCells, M, Sf, bFaceMapping)
end

function kernel_DivOnlyPrecalculatedWeightsBatchedFaceAssembly( # TODO -> Gpu Structs
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
    ϕf = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = weights[iFace]                              # get weight of transport variable interpolation 
    # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
    valueUpper = ϕf * weights_f
    valueLower = -ϕf * (1 - weights_f)

    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x


    idx = offsets[iOwner] + relativeToOwner[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x

    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x


    idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x

    return nothing
end

function gpu_DivOnlyHardcodedDivBatchedFaceAssemblyRunner(
    iOwners::CuArray{Int32},
    iNeighbors::CuArray{Int32},
    gDiffs::CuArray{P},
    offsets::CuArray{Int32},
    nu_g::CuArray{P},
    rows::CuArray{Int32},
    cols::CuArray{Int32},
    vals::CuArray{P},
    entriesNeeded::Int32,
    relativeToOwners::CuArray{Int32},
    N::Int32,
    relativeToNbs::CuArray{Int32},
    numBlocks::Int32,
    bFaceValues::CuArray{SVector{3,P}},
    RHS::CuArray{P},
    nCells::Int32,
    M::Int32,
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32},
    div::String
) where {P<:AbstractFloat}
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


function kernel_DivOnlyHardcodedDivCDFBatchedFaceAssembly( # TODO -> Gpu Structs
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
    ϕf = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = centralDifferencing(ϕf)                             # get weight of transport variable interpolation 
    # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
    valueUpper = ϕf * weights_f
    valueLower = -ϕf * (1 - weights_f)

    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x


    idx = offsets[iOwner] + relativeToOwner[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x


    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x


    idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x

    return nothing
end



function kernel_DivOnlyHardcodedDivUpwindBatchedFaceAssembly( # TODO -> Gpu Structs
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
    ϕf = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = upwind(ϕf)                             # get weight of transport variable interpolation 
    # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
    valueUpper = ϕf * weights_f
    valueLower = -ϕf * (1 - weights_f)

    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x


    idx = offsets[iOwner] + relativeToOwner[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x


    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x


    idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x

    return nothing
end


function gpu_DivOnlyDynamicBatchedFaceAssemblyRunner(
    iOwners::CuArray{Int32},
    iNeighbors::CuArray{Int32},
    gDiffs::CuArray{P},
    offsets::CuArray{Int32},
    nu_g::CuArray{P},
    rows::CuArray{Int32},
    cols::CuArray{Int32},
    vals::CuArray{P},
    entriesNeeded::Int32,
    relativeToOwners::CuArray{Int32},
    N::Int32,
    relativeToNbs::CuArray{Int32},
    numBlocks::Int32,
    bFaceValues::CuArray{SVector{3,P}},
    RHS::CuArray{P},
    nCells::Int32,
    M::Int32,
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32},
    div::Function
) where {P<:AbstractFloat}
    for color in 1:numBatches
        iblocks = cld(N, 256)
        CUDA.@sync @cuda threads = 256 blocks = iblocks kernel_DivOnlyDynamicBatchedFaceAssembly(iOwners, iNeighbors, offsets, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, faceColorMapping, color, U, Sf, div)
    end
    bblocks = cld(M, 256)

    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace_DivOnly(iOwners, bFaceValues, RHS, N, nCells, M, Sf, bFaceMapping)
end

function kernel_DivOnlyDynamicBatchedFaceAssembly( # TODO -> Gpu Structs
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
    ϕf = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = div(ϕf)                             # get weight of transport variable interpolation 
    # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
    valueUpper = ϕf * weights_f
    valueLower = -ϕf * (1 - weights_f)

    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x


    idx = offsets[iOwner] + relativeToOwner[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x


    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x


    idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x

    return nothing
end

function gpu_PrecalculatedWeightsBatchedFaceAssemblyRunner(
    iOwners::CuArray{Int32},
    iNeighbors::CuArray{Int32},
    gDiffs::CuArray{P},
    offsets::CuArray{Int32},
    nu_g::CuArray{P},
    rows::CuArray{Int32},
    cols::CuArray{Int32},
    vals::CuArray{P},
    entriesNeeded::Int32,
    relativeToOwners::CuArray{Int32},
    N::Int32,
    relativeToNbs::CuArray{Int32},
    numBlocks::Int32,
    bFaceValues::CuArray{SVector{3,P}},
    RHS::CuArray{P},
    nCells::Int32,
    M::Int32,
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32},
    weights::CuArray{P}
) where {P<:AbstractFloat}
    for color in 1:numBatches
        iblocks = cld(N, 256)
        CUDA.@sync @cuda threads = 256 blocks = iblocks kernel_PrecalculatedWeightsBatchedFaceAssembly(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, faceColorMapping, color, U, Sf, weights)
    end
    bblocks = cld(M, 256)

    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace(iOwners, gDiffs, offsets, nu_g, vals, bFaceValues, RHS, N, nCells, M, Sf, bFaceMapping)
end

function kernel_PrecalculatedWeightsBatchedFaceAssembly( # TODO -> Gpu Structs
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
    ϕf = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = weights[iFace]                              # get weight of transport variable interpolation 
    # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
    valueUpper = ϕf * weights_f + diffusion
    valueLower = -ϕf * (1 - weights_f) - diffusion

    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x


    idx = offsets[iOwner] + relativeToOwner[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x


    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x


    idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x

    return nothing
end


function gpu_HardcodedBatchedFaceAssemblyRunner(iOwners::CuArray{Int32},
    iNeighbors::CuArray{Int32},
    gDiffs::CuArray{P},
    offsets::CuArray{Int32},
    nu_g::CuArray{P},
    rows::CuArray{Int32},
    cols::CuArray{Int32},
    vals::CuArray{P},
    entriesNeeded::Int32,
    relativeToOwners::CuArray{Int32},
    N::Int32,
    relativeToNbs::CuArray{Int32},
    numBlocks::Int32,
    bFaceValues::CuArray{SVector{3,P}},
    RHS::CuArray{P},
    nCells::Int32,
    M::Int32,
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32},
    div::String
) where {P<:AbstractFloat}
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

    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace(iOwners, gDiffs, offsets, nu_g, vals, bFaceValues, RHS, N, nCells, M, Sf, bFaceMapping)
end

function kernel_HardcodedCDFBatchedFaceAssembly( # TODO -> Gpu Structs
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
    ϕf = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = centralDifferencing(ϕf)                              # get weight of transport variable interpolation 
    # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
    valueUpper = ϕf * weights_f + diffusion
    valueLower = -ϕf * (1 - weights_f) - diffusion

    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x


    idx = offsets[iOwner] + relativeToOwner[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x


    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x

    idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x

    return nothing
end


function kernel_HardcodedUpwindBatchedFaceAssembly( # TODO -> Gpu Structs
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
    ϕf = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = upwind(ϕf)                              # get weight of transport variable interpolation 
    # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
    valueUpper = ϕf * weights_f + diffusion
    valueLower = -ϕf * (1 - weights_f) - diffusion

    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x


    idx = offsets[iOwner] + relativeToOwner[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x


    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x


    idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x

    return nothing
end
function gpu_DynamicBatchedFaceAssemblyRunner(
    iOwners::CuArray{Int32},
    iNeighbors::CuArray{Int32},
    gDiffs::CuArray{P},
    offsets::CuArray{Int32},
    nu_g::CuArray{P},
    rows::CuArray{Int32},
    cols::CuArray{Int32},
    vals::CuArray{P},
    entriesNeeded::Int32,
    relativeToOwners::CuArray{Int32},
    N::Int32,
    relativeToNbs::CuArray{Int32},
    numBlocks::Int32,
    bFaceValues::CuArray{SVector{3,P}},
    RHS::CuArray{P},
    nCells::Int32,
    M::Int32,
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32},
    div::Function
) where {P<:AbstractFloat}
    for color in 1:numBatches
        iblocks = cld(N, 256)
        CUDA.@sync @cuda threads = 256 blocks = iblocks kernel_DynamicBatchedFaceAssembly(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, relativeToOwners, N, relativeToNbs, faceColorMapping, color, U, Sf, div)
    end
    bblocks = cld(M, 256)

    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace(iOwners, gDiffs, offsets, nu_g, vals, bFaceValues, RHS, N, nCells, M, Sf, bFaceMapping)
end



function kernel_DynamicBatchedFaceAssembly( # TODO -> Gpu Structs
    iOwners,
    iNeighbors,
    gDiffs,
    offsets,
    nus,
    rows,
    cols,
    vals,
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
    ϕf = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = div(ϕf)                              # get weight of transport variable interpolation 
    # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
    valueUpper = ϕf * weights_f + diffusion
    valueLower = -ϕf * (1 - weights_f) - diffusion

    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x

    idx = offsets[iOwner] + relativeToOwner[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x

    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x

    idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x
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
    gDiffs::CuArray{P},
    offsets::CuArray{Int32},
    nu_g::CuArray{P},
    rows::CuArray{Int32},
    cols::CuArray{Int32},
    vals::CuArray{P},
    entriesNeeded::Int32,
    relativeToOwners::CuArray{Int32},
    N::Int32,
    relativeToNbs::CuArray{Int32},
    numBlocks::Int32,
    bFaceValues::CuArray{SVector{3,P}},
    RHS::CuArray{P},
    nCells::Int32,
    M::Int32,
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32},
) where {P<:AbstractFloat}
    for color in 1:numBatches
        iblocks = cld(N, 256)
        CUDA.@sync @cuda threads = 256 blocks = iblocks kernel_internalFace_coloured(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, faceColorMapping, color, U, Sf)
    end
    bblocks = cld(M, 256)

    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace(iOwners, gDiffs, offsets, nu_g, vals, bFaceValues, RHS, N, nCells, M, Sf, bFaceMapping)
end

function kernel_internalFace_coloured( # TODO -> Gpu Structs
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
    ϕf = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = 0.5                              # get weight of transport variable interpolation 
    # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
    valueUpper = ϕf * weights_f + diffusion
    valueLower = -ϕf * (1 - weights_f) - diffusion

    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x


    idx = offsets[iOwner] + relativeToOwner[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x


    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    vals[idx] += valueLower    # x


    idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += valueUpper    # x

    return nothing
end


function fusedBatchedFaceBasedRunner(
    iOwners::CuArray{Int32},
    iNeighbors::CuArray{Int32},
    gDiffs::CuArray{P},
    offsets::CuArray{Int32},
    nu_g::CuArray{P},
    rows::CuArray{Int32},
    cols::CuArray{Int32},
    vals::CuArray{P},
    entriesNeeded::Int32,
    relativeToOwners::CuArray{Int32},
    N::Int32,
    relativeToNbs::CuArray{Int32},
    numBlocks::Int32,
    bFaceValues::CuArray{SVector{3,P}},
    RHS::CuArray{P},
    nCells::Int32,
    M::Int32,
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32},
    ops
) where {P<:AbstractFloat}
    for color in 1:numBatches
        iblocks = cld(N, 256)
        CUDA.@sync @cuda threads = 256 blocks = iblocks fusedKernel_internal_colored(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, relativeToOwners, N, relativeToNbs, faceColorMapping, color, U, Sf, ops)
    end
    bblocks = cld(M, 256)

    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace(iOwners, gDiffs, offsets, nu_g, vals, bFaceValues, RHS, N, nCells, M, Sf, bFaceMapping)
end

function fusedKernel_internal_colored(
    iOwners,
    iNeighbors,
    gDiffs,
    offsets,
    nus,
    rows,
    cols,
    vals,
    relativeToOwners,
    numInteriorFaces,
    relativeToNeighbor,
    faceColors,
    color,
    U,
    Sf,
    ops
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

    upper = 0.0
    lower = 0.0
    upper, lower = ops(U[iOwner], U[iNeighbor], Sf[iFace], nus[iOwner], gDiffs[iFace], upper, lower)

    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    vals[idx] += upper

    idx = offsets[iOwner] + relativeToOwners[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] += lower

    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    vals[idx] += lower

    idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += upper
    return nothing
end


############################# helper 


function gpu_prepareBatchedFaceBased(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, numBlocks, bFaceValues, RHS, nCells, M, U, Sf, bFaceMapping = gpu_prepareFaceBased(input)
    numBatches, faceColorMapping = getGreedyEdgeColoring(input) |> cu
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
    # return maximum(faceColorMapping), CuArray{Int32}(faceColorMapping)
    return maximum(faceColorMapping), faceColorMapping
end



###### provable not really useful




function batchedFaceBasedAllRunner(
    iOwners::CuArray{Int32},
    iNeighbors::CuArray{Int32},
    gDiffs::CuArray{P},
    offsets::CuArray{Int32},
    nu_g::CuArray{P},
    rows::CuArray{Int32},
    cols::CuArray{Int32},
    vals::CuArray{P},
    entriesNeeded::Int32,
    relativeToOwners::CuArray{Int32},
    N::Int32,
    relativeToNbs::CuArray{Int32},
    numBlocks::Int32,
    bFaceValues::CuArray{SVector{3,P}},
    RHS::CuArray{P},
    nCells::Int32,
    M::Int32,
    numBatches::Int32,
    faceColorMapping::CuArray{Int32},
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32}
) where {P<:AbstractFloat}
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
function kernel_all_colored( # TODO -> Gpu Structs # TODO -> Gpu Structs
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
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = 0.5                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = ϕf * weights_f + diffusion
        valueLower = -ϕf * (1 - weights_f) - diffusion

        idx = offsets[iOwner]
        cols[idx] = iOwner
        rows[idx] = iOwner
        vals[idx] += valueUpper    # x


        idx = offsets[iOwner] + relativeToOwner[iFace]
        cols[idx] = iOwner
        rows[idx] = iNeighbor
        vals[idx] += valueLower    # x

        idx = offsets[iNeighbor]
        cols[idx] = iNeighbor
        rows[idx] = iNeighbor
        vals[idx] += valueLower    # x

        idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
        cols[idx] = iNeighbor
        rows[idx] = iOwner
        vals[idx] += valueUpper    # x
    else
        relativeFaceIndex = iFace - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex == -1 # boundarytype != fixed
            return
        end
        convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])
        diffusion = nus[iOwner] * gDiffs[iFace]
        idx = offsets[iOwner]
        vals[idx] -= diffusion[1]    # x

        # RHS/Source
        value = convection .+ diffusion
        RHS[iOwner] -= value[1]
        RHS[iOwner+nCells] -= value[2]
        RHS[iOwner+nCells+nCells] -= value[3]
    end
    return nothing
end

### comparison to KA 

function fusedBatchedFaceBasedRunner(
    faces, # GpuFace[] 
    nus,
    offsets,
    bFaceValues,
    U,
    boundaries,
    rows,
    cols,
    vals,
    RHS,
    numBatches,
    N,
    M,
    fused_pde
)
    for color in 1:numBatches
        iblocks = cld(N, 256)
        CUDA.@sync @cuda threads = 256 blocks = iblocks fusedKernel_internal_colored(
            faces,      # CuArray{GpuFace}
            nus,
            offsets,
            U,
            rows,
            cols,
            vals,
            color,
            fused_pde
        )
    end
    bblocks = cld(M, 256)

    CUDA.@sync @cuda threads = 256 blocks = bblocks kernel_boundaryFace_structs(
        faces,      # CuArray{GpuFace}
        nus,
        offsets,
        bFaceValues,
        boundaries,
        vals,
        RHS,
        fused_pde
    )
    return rows, cols, vals, RHS
end

function fusedKernel_internal_colored(
    faces,      # CuArray{GpuFace}
    nus,
    offsets,
    U,
    rows,
    cols,
    vals,
    color,
    fused_pde
)
    iFace = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if iFace > length(faces)
        return
    end
    theFace = faces[iFace]
    if theFace.iNeighbor == -1
        return
    end
    if theFace.batchId != color
        return
    end
    iOwner = theFace.iOwner
    iNeighbor = theFace.iNeighbor

    upper = 0.0
    lower = 0.0
    upper, lower = fused_pde(U[iOwner], U[iNeighbor], theFace.Sf, nus[iOwner], theFace.gDiff, upper, lower)

    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    vals[idx] += upper

    idx = offsets[iOwner] + theFace.relativeToOwner
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] += lower

    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    vals[idx] += lower

    idx = offsets[iNeighbor] + theFace.relativeToNeighbor
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += upper

    return nothing
end

function kernel_boundaryFace_structs(
    faces,      # CuArray{GpuFace}
    nus,
    offsets,
    bFaceValues,
    boundaries,
    vals,
    RHS,
    fused_pde
)
    iFace = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    globalIndex = iFace + length(bFaceValues)
    theFace = faces[iFace]
    if theFace.iNeighbor != -1
        return
    end
    iOwner = theFace.iOwner
    iBoundary = theFace.patchIndex
    if boundaries[iBoundary].isFixedValue
        diag, rhsx, rhsy, rhsz = fused_pde(bFaceValues[iFace], theFace.Sf, nus[iOwner], theFace.gDiff)

        # Diagonal Entry        
        vals[offsets[iOwner]] += diag

        # RHS/Source
        nCells = length(nus)
        RHS[iOwner] += rhsx
        RHS[iOwner+nCells] += rhsy
        RHS[iOwner+nCells+nCells] += rhsz
    end
    return nothing
end