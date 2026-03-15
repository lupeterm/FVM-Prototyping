include("init.jl")
include("gpu_helper.jl")

function gpu_FaceBasedAssembly(input::MatrixAssemblyInput)
    iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, numBlocks, bFaceValues, RHS, nCells, M, U, Sf, bFaceMapping = gpu_prepareFaceBased(input)
    faceBasedAllRunner(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, numBlocks, bFaceValues, RHS, nCells, M, U, Sf, bFaceMapping)
    return Vector(rows), Vector(cols), Vector(vals), Vector(RHS)
end

# div only precalc 
function gpu_DivOnlyPrecalculatedWeightsFaceBasedAssemblyRunner(
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
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32},
    weights::CuArray{P}
) where {P<:AbstractFloat}
    CUDA.@sync @cuda threads = 256 blocks = numBlocks kernel_DivOnlyPrecalculatedWeightsFaceBasedAssembly(
        iOwners,
        iNeighbors,
        offsets,
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
        bFaceMapping,
        weights
    )
    return rows, cols, vals, RHS
end

function kernel_DivOnlyPrecalculatedWeightsFaceBasedAssembly(
    iOwners,
    iNeighbors,
    offsets,
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
    bFaceMapping,
    weights
)
    iFace = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if iFace > numInternalFaces + numBoundaryFaces
        return
    end

    iOwner = iOwners[iFace]
    if iFace <= numInternalFaces

        iNeighbor = iNeighbors[iFace]
        # Diffusion

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf::P = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = weights[iFace]                          # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::P = ϕf * weights_f
        valueLower::P = -ϕf * (1 - weights_f)

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
        # RHS/Source
        CUDA.@atomic RHS[iOwner] -= convection[1]
        CUDA.@atomic RHS[iOwner+nCells] -= convection[2]
        CUDA.@atomic RHS[iOwner+nCells+nCells] -= convection[3]
    end
    return nothing
end


# div only hardcoded upwind
function gpu_DivOnlyHardcodedDivFaceBasedAssemblyRunner(
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
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32},
    upwindOrCdf::String
) where {P<:AbstractFloat}
    if upwindOrCdf == "CDF"
        CUDA.@sync @cuda threads = 256 blocks = numBlocks kernel_DivOnlyHardcodedUpwindFaceBasedAssembly(
            iOwners,
            iNeighbors,
            offsets,
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
            bFaceMapping,
        )
    else
        CUDA.@sync @cuda threads = 256 blocks = numBlocks kernel_DivOnlyHardcodedCDFFaceBasedAssembly(
            iOwners,
            iNeighbors,
            offsets,
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
            bFaceMapping,
        )
    end
    return rows, cols, vals, RHS
end


function kernel_DivOnlyHardcodedUpwindFaceBasedAssembly(
    iOwners,
    iNeighbors,
    offsets,
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

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf::P = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = upwind(ϕf)                          # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::P = ϕf * weights_f
        valueLower::P = -ϕf * (1 - weights_f)

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
        # RHS/Source
        CUDA.@atomic RHS[iOwner] -= convection[1]
        CUDA.@atomic RHS[iOwner+nCells] -= convection[2]
        CUDA.@atomic RHS[iOwner+nCells+nCells] -= convection[3]
    end
    return nothing
end


function kernel_DivOnlyHardcodedCDFFaceBasedAssembly(
    iOwners,
    iNeighbors,
    offsets,
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

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf::P = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = centralDifferencing(ϕf)                          # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::P = ϕf * weights_f
        valueLower::P = -ϕf * (1 - weights_f)

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
        # RHS/Source
        CUDA.@atomic RHS[iOwner] -= convection[1]
        CUDA.@atomic RHS[iOwner+nCells] -= convection[2]
        CUDA.@atomic RHS[iOwner+nCells+nCells] -= convection[3]
    end
    return nothing
end


function gpu_DivOnlyDynamicFaceBasedAssemblyRunner(
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
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32},
    divFunc::Function
) where {P<:AbstractFloat}
    CUDA.@sync @cuda threads = 256 blocks = numBlocks kernel_DivOnlyDynamicFaceBasedAssembly(
        iOwners,
        iNeighbors,
        offsets,
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
        bFaceMapping,
        divFunc
    )
    return rows, cols, vals, RHS
end

function kernel_DivOnlyDynamicFaceBasedAssembly(
    iOwners,
    iNeighbors,
    offsets,
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
    bFaceMapping,
    divFunc
)
    iFace = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if iFace > numInternalFaces + numBoundaryFaces
        return
    end

    iOwner = iOwners[iFace]
    if iFace <= numInternalFaces

        iNeighbor = iNeighbors[iFace]
        # Diffusion

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf::P = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = divFunc(ϕf)                          # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::P = ϕf * weights_f
        valueLower::P = -ϕf * (1 - weights_f)

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
        # RHS/Source
        CUDA.@atomic RHS[iOwner] -= convection[1]
        CUDA.@atomic RHS[iOwner+nCells] -= convection[2]
        CUDA.@atomic RHS[iOwner+nCells+nCells] -= convection[3]
    end
    return nothing
end

function gpu_PrecalculatedWeightsFaceBasedAssemblyRunner(
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
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32},
    weights::CuArray{P}
) where {P<:AbstractFloat}
    CUDA.@sync @cuda threads = 256 blocks = numBlocks kernel_PrecalculatedWeightsFaceBasedAssembly(
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
        bFaceMapping,
        weights
    )
    return rows, cols, vals, RHS
end

function kernel_PrecalculatedWeightsFaceBasedAssembly(
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
    bFaceMapping,
    weights
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
        ϕf::P = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = weights[iFace]                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::P = ϕf * weights_f + diffusion
        valueLower::P = -ϕf * (1 - weights_f) - diffusion

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

function gpu_HardcodedFaceBasedAssemblyRunner(
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
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32},
    upwindOrCdf::String
) where {P<:AbstractFloat}
    if upwindOrCdf == "CDF"
        CUDA.@sync @cuda threads = 256 blocks = numBlocks kernel_HardcodedUpwindFaceBasedAssembly(
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
    else
        CUDA.@sync @cuda threads = 256 blocks = numBlocks kernel_HardcodedCDFFaceBasedAssembly(
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
    return rows, cols, vals, RHS
end

function kernel_HardcodedUpwindFaceBasedAssembly(
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
        ϕf::P = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = upwind(ϕf)                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::P = ϕf * weights_f + diffusion
        valueLower::P = -ϕf * (1 - weights_f) - diffusion

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


function kernel_HardcodedCDFFaceBasedAssembly(
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
        ϕf::P = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = centralDifferencing(ϕf)                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::P = ϕf * weights_f + diffusion
        valueLower::P = -ϕf * (1 - weights_f) - diffusion

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

function gpu_DynamicFaceBasedAssemblyRunner(
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
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32},
    divFunc::Function
) where {P<:AbstractFloat}
    CUDA.@sync @cuda threads = 256 blocks = numBlocks kernel_DynamicFaceBasedAssembly(
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
        bFaceMapping,
        divFunc
    )
end

function kernel_DynamicFaceBasedAssembly(
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
    bFaceMapping,
    divFunc
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
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = divFunc(ϕf)                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = ϕf * weights_f + diffusion
        valueLower = -ϕf * (1 - weights_f) - diffusion

        idx = offsets[iOwner]
        cols[idx] = iOwner
        rows[idx] = iOwner
        CUDA.@atomic vals[idx] += valueUpper    # x

        idx = offsets[iOwner] + relativeToOwner[iFace]
        cols[idx] = iOwner
        rows[idx] = iNeighbor
        vals[idx] += valueLower    # x

        idx = offsets[iNeighbor]
        cols[idx] = iNeighbor
        rows[idx] = iNeighbor
        CUDA.@atomic vals[idx] += valueLower    # x

        idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
        cols[idx] = iNeighbor
        rows[idx] = iOwner
        vals[idx] += valueUpper    # x
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

        # RHS/Source
        value = convection .+ diffusion
        CUDA.@atomic RHS[iOwner] -= value[1]
        CUDA.@atomic RHS[iOwner+nCells] -= value[2]
        CUDA.@atomic RHS[iOwner+nCells+nCells] -= value[3]
    end
    return nothing
end



function gpu_LaplaceOnlyFaceBasedAssemblyRunner(
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
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32}
) where {P<:AbstractFloat}
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
    return rows, cols, vals, RHS
end

function kernel_LaplaceOnlyFaceBasedAssembly(
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

        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::P = diffusion
        valueLower::P = -diffusion

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
        diffusion = bFaceValues[bFaceIndex] .* nus[iOwner] * gDiffs[iFace]
        idx = offsets[iOwner]
        CUDA.@atomic vals[idx] -= diffusion[1]    # x
        CUDA.@atomic vals[idx+entriesNeeded] -= diffusion[2]  # y
        CUDA.@atomic vals[idx+entriesNeeded+entriesNeeded] -= diffusion[3]  # z

        # RHS/Source
        value = diffusion
        CUDA.@atomic RHS[iOwner] -= value[1]
        CUDA.@atomic RHS[iOwner+nCells] -= value[2]
        CUDA.@atomic RHS[iOwner+nCells+nCells] -= value[3]
    end
    return nothing
end

function faceBasedAllRunner(
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
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32}
) where {P<:AbstractFloat}
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
        ϕf::P = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = 0.5                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::P = ϕf * weights_f + diffusion
        valueLower::P = -ϕf * (1 - weights_f) - diffusion

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
    bblocks::Int32,
    bFaceValues::CuArray{SVector{3,P}},
    RHS::CuArray{P},
    nCells::Int32,
    M::Int32,
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32}
) where {P<:AbstractFloat}
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
    ϕf::P = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = upwind(ϕf)                              # get weight of transport variable interpolation 
    # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
    valueUpper::P = ϕf * weights_f + diffusion
    valueLower::P = -ϕf * (1 - weights_f) - diffusion

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

    # RHS/Source
    value = convection .+ diffusion
    CUDA.@atomic RHS[iOwner] -= value[1]
    CUDA.@atomic RHS[iOwner+nCells] -= value[2]
    CUDA.@atomic RHS[iOwner+nCells+nCells] -= value[3]
    return nothing
end


function kernel_boundaryFace_LaplaceOnly(
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
    diffusion = bFaceValues[bFaceIndex] .* nus[iOwner] * gDiffs[globalFaceIndex]
    idx = offsets[iOwner]
    CUDA.@atomic vals[idx] -= diffusion[1]    # x
    CUDA.@atomic vals[idx+entriesNeeded] -= diffusion[2]  # y
    CUDA.@atomic vals[idx+entriesNeeded+entriesNeeded] -= diffusion[3]  # z

    # RHS/Source
    CUDA.@atomic RHS[iOwner] -= diffusion[1]
    CUDA.@atomic RHS[iOwner+nCells] -= diffusion[2]
    CUDA.@atomic RHS[iOwner+nCells+nCells] -= diffusion[3]
    return nothing
end


function kernel_boundaryFace_DivOnly(
    iOwners,
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

    # RHS/Source
    CUDA.@atomic RHS[iOwner] -= convection[1]
    CUDA.@atomic RHS[iOwner+nCells] -= convection[2]
    CUDA.@atomic RHS[iOwner+nCells+nCells] -= convection[3]
    return nothing
end
