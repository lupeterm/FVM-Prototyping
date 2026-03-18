include("init.jl")
include("gpu_faceBased.jl")
include("gpu_helper.jl")
using KernelAbstractions, Atomix
"""
'abstract', as in, using `KernelAbstractions`, and not CUDA.
"""

function run_faceBased_abstract(
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
    bFaceValues::CuArray{SVector{3,P}},
    RHS::CuArray{P},
    nCells::Int32,
    numBoundaryFaces::Int32,
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32},
    ops,
) where {P<:AbstractFloat}
    backend = CUDABackend()
    kernel_FusedFaceBasedAssembly(backend, 256)(
        iOwners,
        iNeighbors,
        gDiffs,
        offsets,
        nu_g,
        rows,
        cols,
        vals,
        RHS,
        entriesNeeded,
        relativeToOwners,
        N,
        relativeToNbs,
        U,
        Sf,
        bFaceMapping,
        bFaceValues,
        numBoundaryFaces,
        nCells,
        ops;
        ndrange=N + numBoundaryFaces
    )
    KernelAbstractions.synchronize(backend)
    return rows, cols, vals, RHS
end


function run_faceBased_abstract_mtl(
    iOwners,
    iNeighbors,
    gDiffs,
    offsets,
    nu_g,
    rows,
    cols,
    vals,
    RHS,
    entriesNeeded,
    relativeToOwners,
    N,
    relativeToNbs,
    U,
    Sf,
    bFaceMapping,
    bFaceValues,
    numBoundaryFaces,
    nCells,
    ops,
) 
    backend = get_backend(iOwners)
    kernel_FusedFaceBasedAssembly(backend, 32)(
        iOwners,
        iNeighbors,
        gDiffs,
        offsets,
        nu_g,
        rows,
        cols,
        vals,
        RHS,
        entriesNeeded,
        relativeToOwners,
        N,
        relativeToNbs,
        U,
        Sf,
        bFaceMapping,
        bFaceValues,
        numBoundaryFaces,
        nCells,
        ops;
        ndrange=N
    )
    KernelAbstractions.synchronize(backend)
    # return rows, cols, vals, RHS
end

@kernel function kernel_FusedFaceBasedAssembly(
    @Const(iOwners),
    @Const(iNeighbors),
    @Const(gDiffs),
    @Const(offsets),
    @Const(nus),
    rows,
    cols,
    vals,
    RHS,
    @Const(entriesNeeded),
    @Const(relativeToOwner),
    @Const(numInteriorFaces),
    @Const(relativeToNeighbor),
    @Const(U),
    @Const(Sf),
    @Const(bFaceMapping),
    @Const(bFaceValues),
    @Const(numBoundaryFaces),
    @Const(nCells),
    ops,
)
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    if iFace <= numInteriorFaces

        upper = 0.0
        lower = 0.0

        upper, lower = ops(U[iOwner], U[iNeighbor], Sf[iFace], nus[iFace], gDiffs[iFace], upper, lower)

        idx = offsets[iOwner]
        cols[idx] = iOwner
        rows[idx] = iOwner
        Atomix.@atomic vals[idx] += upper

        idx = offsets[iOwner] + relativeToOwner[iFace]
        cols[idx] = iOwner
        rows[idx] = iNeighbor
        vals[idx] += lower

        idx = offsets[iNeighbor]
        cols[idx] = iNeighbor
        rows[idx] = iNeighbor
        Atomix.@atomic vals[idx] += lower

        idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
        cols[idx] = iNeighbor
        rows[idx] = iOwner
        vals[idx] += upper
    else
        relativeFaceIndex = iFace - numInteriorFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])
            diffusion = nus[iOwner] * gDiffs[iFace]
            Atomix.@atomic vals[idx] -= diffusion[1] # FIXME local scope in julia is a myth 

            # RHS/Source
            Atomix.@atomic RHS[iOwner] -= convection[1] - diffusion
            Atomix.@atomic RHS[iOwner+nCells] -= convection[2] - diffusion
            Atomix.@atomic RHS[iOwner+nCells+nCells] -= convection[3] - diffusion
        end
    end
end


function run_batchedFace_abstract(
    iOwners::CuArray{Int32},
    iNeighbors::CuArray{Int32},
    gDiffs::CuArray{P},
    offsets::CuArray{Int32},
    nus::CuArray{P},
    rows::CuArray{Int32},
    cols::CuArray{Int32},
    vals::CuArray{P},
    entriesNeeded::Int32,
    relativeToOwners::CuArray{Int32},
    N::Int32,
    relativeToNeighbor::CuArray{Int32},
    numBlocks::Int32,
    bFaceValues::CuArray{SVector{3,P}},
    RHS::CuArray{P},
    nCells::Int32,
    M::Int32,
    numBatches::Int32,
    faceColors::CuArray{Int32},
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32},
    ops,
) where {P<:AbstractFloat}
    backend = CUDABackend()
    for color in 1:numBatches
        kernel_FusedBatchedFaceAssembly(backend, 256)(
            iOwners,
            iNeighbors,
            gDiffs,
            offsets,
            nus,
            rows,
            cols,
            vals,
            relativeToOwners,
            relativeToNeighbor,
            U,
            Sf,
            faceColors,
            color,
            ops;
            ndrange=N
        )
        KernelAbstractions.synchronize(backend)
    end
    abstract_boundaryFace(backend, 256)(
        iOwners,
        gDiffs,
        offsets,
        nus,
        vals,
        bFaceValues,
        RHS,
        N,
        nCells,
        M,
        Sf,
        bFaceMapping;
        ndrange=M
    )
    KernelAbstractions.synchronize(backend)

    return rows, cols, vals, RHS
end


@kernel function kernel_FusedBatchedFaceAssembly(
    iOwners,
    iNeighbors,
    gDiffs,
    offsets,
    nus,
    rows,
    cols,
    vals,
    relativeToOwners,
    relativeToNeighbor,
    U,
    Sf,
    faceColors,
    color,
    ops, # TODO: differentiate between internal and boundary
)
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    faceColor = faceColors[iFace]

    if faceColor == color
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
    end
end


@kernel function abstract_boundaryFace(
    @Const(iOwners),
    @Const(gDiffs),
    @Const(offsets),
    @Const(nus),
    vals,
    @Const(bFaceValues),
    RHS,
    @Const(numInternalFaces),
    @Const(nCells),
    @Const(numBoundaryFaces),
    @Const(Sf),
    @Const(bFaceMapping)
)
    iFace = @index(Global)
    globalFaceIndex = numInternalFaces + iFace
    bFaceIndex = bFaceMapping[iFace]
    if bFaceIndex != -1
        iOwner = iOwners[globalFaceIndex]
        convection = bFaceValues[bFaceIndex] .* dot(Sf[globalFaceIndex], bFaceValues[bFaceIndex])
        diffusion = nus[iOwner] * gDiffs[globalFaceIndex]
        idx = offsets[iOwner]
        Atomix.@atomic vals[idx] -= diffusion[1]    # x

        # RHS/Source
        Atomix.@atomic RHS[iOwner] -= convection[1] - diffusion
        Atomix.@atomic RHS[iOwner+nCells] -= convection[2] - diffusion
        Atomix.@atomic RHS[iOwner+nCells+nCells] -= convection[3] - diffusion
    end
end
