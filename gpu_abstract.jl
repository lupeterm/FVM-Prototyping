include("init.jl")
include("gpu_faceBased.jl")
include("gpu_helper.jl")
using KernelAbstractions, Atomix

function cuda_DynamicBatchedFaceAssemblyRunner(
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
    backend = get_backend(iOwners)
    for color in 1:numBatches
        ev = abstractAssembly(backend)(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, faceColorMapping, color, U, Sf, div; ndrange=N)
        KernelAbstractions.synchronize(backend)
    end
    ev = boundary(backend, 1)(iOwners, gDiffs, offsets, nu_g, vals, entriesNeeded, bFaceValues, RHS, N, nCells, M, Sf, bFaceMapping; ndrange=M)
    KernelAbstractions.synchronize(backend)
end

@kernel function boundary(
    @Const(iOwners),
    @Const(gDiffs),
    @Const(offsets),
    @Const(nus),
    vals,
    @Const(entriesNeeded),
    @Const(bFaceValues),
    RHS,
    @Const(numInternalFaces),
    @Const(nCells),
    @Const(numBoundaryFaces),
    @Const(Sf),
    @Const(bFaceMapping)
)
    # iFace = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iFace = @index(Global)
    globalFaceIndex = numInternalFaces + iFace
    bFaceIndex = bFaceMapping[iFace]
    if bFaceIndex != -1
        iOwner = iOwners[globalFaceIndex]
        convection = bFaceValues[bFaceIndex] .* dot(Sf[globalFaceIndex], bFaceValues[bFaceIndex])
        diffusion = bFaceValues[bFaceIndex] .* nus[iOwner] * gDiffs[globalFaceIndex]
        idx = offsets[iOwner]
        Atomix.@atomic vals[idx] -= diffusion[1]    # x
        
        # RHS/Source
        value = convection .+ diffusion
        Atomix.@atomic RHS[iOwner] -= value[1]
        Atomix.@atomic RHS[iOwner+nCells] -= value[2]
        Atomix.@atomic RHS[iOwner+nCells+nCells] -= value[3]
    end
end

@kernel function abstractAssembly(
    @Const(iOwners),
    @Const(iNeighbors),
    @Const(gDiffs),
    @Const(offsets),
    @Const(nus),
    rows,
    cols,
    vals,
    @Const(entriesNeeded),
    @Const(relativeToOwner),
    @Const(numInteriorFaces),
    @Const(relativeToNeighbor),
    @Const(faceColors),
    @Const(color),
    @Const(U),
    @Const(Sf),
    div
)
    # iFace = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iFace = @index(Global)
    faceColor = faceColors[iFace]
    if faceColor == color
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
        Atomix.@atomic vals[idx] += valueUpper    # x

        idx = offsets[iOwner] + relativeToOwner[iFace]
        cols[idx] = iOwner
        rows[idx] = iNeighbor
        vals[idx] += valueLower    # x

        idx = offsets[iNeighbor]
        cols[idx] = iNeighbor
        rows[idx] = iNeighbor
        Atomix.@atomic vals[idx] += valueLower    # x

        idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
        cols[idx] = iNeighbor
        rows[idx] = iOwner
        vals[idx] += valueUpper    # x
        # return nothing
    end
end


@kernel function test(ops::Vector{FVMOP{T}}) where {T<:AbstractFloat}
    @inbounds for i in eachindex(ops)
        @print("$(String(Symbol(ops[i].scheme)))")
    end
end