include("init.jl")
include("gpu_faceBased.jl")
include("gpu_helper.jl")
include("gpu_batchedFace.jl")
using KernelAbstractions, Atomix, SplitApplyCombine
"""
'abstract', as in, using `KernelAbstractions`, and not CUDA.
"""
function getFaceBasedGpuInput(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu_g = CuArray(input.nu)
    nCells::Int32 = length(mesh.cells)
    RHS = CUDA.zeros(P, nCells * 3)
    vals = CUDA.zeros(P, entriesNeeded)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = gpu_precalcOffsets(input)
    rows = CUDA.zeros(Int32, entriesNeeded)
    cols = CUDA.zeros(Int32, entriesNeeded)
    prepareRelativeIndices!(input)
    bFaceValues, U, bFaceMapping = gpu_getFaceValues(input)
    iOwners, iNeighbors, gDiffs, relativeToOwners, relativeToNbs, Sf = facesToGPUarrays(mesh.faces)
    return iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, relativeToOwners, relativeToNbs, U, Sf, bFaceValues, RHS, input.mesh.numInteriorFaces, bFaceMapping
end

# function run_faceBased_abstract(
#     iOwners,
#     iNeighbors,
#     gDiffs,
#     offsets,
#     nus,
#     rows,
#     cols,
#     vals,
#     relativeToOwner,
#     relativeToNeighbor,
#     U,
#     Sf,
#     bFaceValues,
#     RHS,
#     numInternalFaces,
#     bFaceMapping,
#     fused_pde
# )
#     backend = get_backend(iOwners)
#     kernel_FusedFaceBasedAssembly(backend, 256)(
#         iOwners,
#         iNeighbors,
#         gDiffs,
#         offsets,
#         nus,
#         rows,
#         cols,
#         vals,
#         relativeToOwner,
#         relativeToNeighbor,
#         U,
#         Sf,
#         bFaceValues,
#         RHS,
#         numInternalFaces,
#         bFaceMapping,
#         fused_pde;
#         ndrange=length(iOwners)
#     )
#     KernelAbstractions.synchronize(backend)
#     return rows, cols, vals, RHS
# end

# @kernel function kernel_FusedFaceBasedAssembly(
#     @Const(iOwners),
#     @Const(iNeighbors),
#     @Const(gDiffs),
#     @Const(offsets),
#     @Const(nus),
#     rows,
#     cols,
#     vals,
#     @Const(relativeToOwner),
#     @Const(relativeToNeighbor),
#     @Const(U),
#     @Const(Sf),
#     @Const(bFaceValues),
#     RHS,
#     @Const(numInternalFaces),
#     @Const(bFaceMapping),
#     @Const(fused_pde)
# )
#     iFace = @index(Global)
#     iOwner = iOwners[iFace]
#     iNeighbor = iNeighbors[iFace]
#     if iNeighbor > 0
#         upper, lower = 0.0, 0.0
#         upper, lower = fused_pde(U[iOwner], U[iNeighbor], Sf[iFace], nus[iOwner], gDiffs[iFace], upper, lower)

#         idx = offsets[iOwner]
#         cols[idx] = iOwner
#         rows[idx] = iOwner
#         Atomix.@atomic vals[idx] += upper

#         idx = offsets[iOwner] + relativeToOwner[iFace]
#         cols[idx] = iOwner
#         rows[idx] = iNeighbor
#         vals[idx] += lower
#         idx = offsets[iNeighbor]
#         cols[idx] = iNeighbor
#         rows[idx] = iNeighbor
#         Atomix.@atomic vals[idx] += lower

#         idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
#         cols[idx] = iNeighbor
#         rows[idx] = iOwner
#         vals[idx] += upper
#     else
#         relativeFaceIndex = iFace - numInternalFaces
#         bFaceIndex = bFaceMapping[relativeFaceIndex]
#         if bFaceIndex != -1
#             diag, rhsx, rhsy, rhsz = 0.0, 0.0, 0.0, 0.0
#             diag, rhsx, rhsy, rhsz = fused_pde(bFaceValues[bFaceIndex], Sf[iFace], nus[iOwner], gDiffs[iFace], diag, rhsx, rhsy, rhsz)
#             # Diagonal Entry     
#             idx = offsets[iOwner]
#             CUDA.@atomic vals[idx] += diag

#             # RHS/Source
#             nCells = length(nus)
#             CUDA.@atomic RHS[iOwner] += rhsx
#             CUDA.@atomic RHS[iOwner+nCells] += rhsy
#             CUDA.@atomic RHS[iOwner+nCells+nCells] += rhsz
#         end
#     end
# end

function getBatchedFaceBasedGpuInput(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    numBatches, faceColors = getGreedyEdgeColoring(input) |> cu
    iOwners, iNeighbors, gDiffs, offsets, nus, rows, cols, vals, relativeToOwners, relativeToNbs, U, Sf, bFaceValues, RHS, N, bFaceMapping = getFaceBasedGpuInput(input)
    return iOwners, iNeighbors, gDiffs, offsets, nus, rows, cols, vals, relativeToOwners, relativeToNbs, U, Sf, bFaceValues, RHS, N, bFaceMapping, numBatches, input.mesh.numBoundaryFaces, faceColors
end

# function run_batchedFace_abstract(
#     iOwners,
#     iNeighbors,
#     gDiffs,
#     offsets,
#     nus,
#     rows,
#     cols,
#     vals,
#     relativeToOwners,
#     relativeToNbs,
#     U,
#     Sf,
#     bFaceValues,
#     RHS,
#     N,
#     bFaceMapping,
#     numBatches,
#     M,
#     faceColors,
#     fused_pde
# )
#     backend = CUDABackend()
#     for color in 1:numBatches
#         kernel_FusedBatchedFaceAssembly(backend, 256)(
#             iOwners,
#             iNeighbors,
#             gDiffs,
#             offsets,
#             nus,
#             rows,
#             cols,
#             vals,
#             relativeToOwners,
#             relativeToNbs,
#             faceColors,
#             color,
#             U,
#             Sf,
#             fused_pde;
#             ndrange=N
#         )
#         KernelAbstractions.synchronize(backend)
#     end
#     kernel_abstract_boundaryFace(backend, 256)(
#         iOwners,
#         gDiffs,
#         offsets,
#         nus,
#         vals,
#         bFaceValues,
#         RHS,
#         Sf,
#         bFaceMapping,
#         fused_pde,
#         ndrange=M
#     )
#     KernelAbstractions.synchronize(backend)

#     return rows, cols, vals, RHS
# end


@kernel function kernel_FusedBatchedFaceAssembly(
    @Const(iOwners),
    @Const(iNeighbors),
    @Const(gDiffs),
    @Const(offsets),
    @Const(nus),
    rows,
    cols,
    vals,
    @Const(relativeToOwner),
    @Const(relativeToNeighbor),
    @Const(faceColors),
    @Const(color),
    @Const(U),
    @Const(Sf),
    @Const(fused_pde)
)
    t = typeof(nus[1])
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]

    if faceColors[iFace] == color[1]  # CuArray{Int32, 1, 1}
        upper, lower = fused_pde(U[iOwner], U[iNeighbor], Sf[iFace], nus[iOwner], gDiffs[iFace], zero(t), zero(t))

        idx = offsets[iOwner]
        vals[idx] += upper

        idx = offsets[iOwner] + relativeToOwner[iFace]
        vals[idx] += lower

        idx = offsets[iNeighbor]
        vals[idx] += lower

        idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
        vals[idx] += upper
    end
end


@kernel function kernel_abstract_boundaryFace(
    @Const(iOwners),
    @Const(gDiffs),
    @Const(offsets),
    @Const(nus),
    vals,
    @Const(bFaceValues),
    RHS,
    @Const(Sf),
    @Const(bFaceMapping),
    @Const(fused_pde)
)
    t = typeof(nus[1])
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    bFaceIndex = bFaceMapping[iFace]
    if bFaceIndex != -1
        diag, rhsx, rhsy, rhsz = zero(t), zero(t), zero(t), zero(t)
        diag, rhsx, rhsy, rhsz = fused_pde(bFaceValues[bFaceIndex], Sf[iFace], nus[iOwner], gDiffs[iFace], diag, rhsx, rhsy, rhsz)

        # Diagonal Entry        
        idx = offsets[iOwner]
        Atomix.@atomic vals[idx] += diag

        # RHS/Source
        nCells = length(nus)
        Atomix.@atomic RHS[iOwner] += rhsx
        Atomix.@atomic RHS[iOwner+nCells] += rhsy
        Atomix.@atomic RHS[iOwner+nCells+nCells] += rhsz
    end
end



function getCellBasedGpuInput(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    nus = CuArray(input.nu)
    prepareRelativeIndices!(input)
    # iOwners, _, gDiffs, relativeToOwners, relativeToNbs, Sf = facesToGPUarrays(input.mesh.faces)
    gpuFaces = toGPUSOAs(VectorToSOAs(input.mesh.faces))
    iFaces, iNeighbors, numInts, iFaceOffsets, numFaces = flattenCells(input)
    bFaceValues, U, bFaceMapping = gpu_getFaceValues(input)
    nCells::Int32 = length(input.mesh.cells)
    RHS = CUDA.zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(input.mesh.cells) + 2 * input.mesh.numInteriorFaces
    vals = CUDA.zeros(P, entriesNeeded)
    return iFaces, iNeighbors, numInts, iFaceOffsets, numFaces, nus, gpuFaces.Sf, gpuFaces.gDiffs, U, relativeToOwners, relativeToNbs, bFaceValues, bFaceMapping, input.mesh.numInteriorFaces, gpuFaces.iOwner, vals, RHS
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


function CellInput(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    gpuFaces = toGPUSOAs(VectorToSOAs(input.mesh.faces))

    iFaces, iNeighbors, numInteriors, iFaceOffsets, facesPerCell = flattenCells(input)
    nus = CuArray(input.nu)
    bFaceValues, U, bFaceMapping = gpu_getFaceValues(input)
    rowOffsets = input.offsets |> cu
    nCells::Int32 = length(input.mesh.cells)
    RHS = CUDA.zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(input.mesh.cells) + 2 * input.mesh.numInteriorFaces
    vals = CUDA.zeros(P, entriesNeeded)

    return iFaces, iNeighbors, numInteriors, iFaceOffsets, facesPerCell, nus, gpuFaces.Sf, gpuFaces.gDiff, U, rowOffsets, gpuFaces.ownerRelOwnerIdx, gpuFaces.neighborRelNeighborIdx, bFaceValues, bFaceMapping, gpuFaces.iOwner, vals, RHS
end

function CellBased(args, pde, wg, nd)
    backend = CUDABackend()
    CellBasedKernel(backend, wg)(args..., pde; ndrange=nd)
    KernelAbstractions.synchronize(backend)
    return args[end-1:end]
end

@kernel function CellBasedKernel(
    @Const(iFaces),
    @Const(iNeighbors),
    @Const(numInteriors),
    @Const(iFaceOffsets),
    @Const(facesPerCell),
    @Const(nus),
    @Const(Sf),
    @Const(gDiffs),
    @Const(U),
    @Const(rowOffsets),
    @Const(ownerRelOwnerIdx),
    @Const(neighborRelNeighborIdx),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(iOwners),
    vals,
    RHS,
    @Const(fused_pde)
)
    iElement = @index(Global)
    t = eltype(nus)
    nCells = length(nus)
    numInternalFaces = length(Sf) - length(bFaceMapping)
    if iElement <= nCells
        diag, rhsx, rhsy, rhsz = zero(t), zero(t), zero(t), zero(t)
        @inbounds startIndex = iFaceOffsets[iElement] - Int32(1)
        @inbounds for iFace in one(Int32):numInteriors[iElement]
            @inbounds iFaceIndex = iFaces[startIndex+iFace]
            @inbounds isOwner = iOwners[iFaceIndex] == iElement
            @inbounds valueUpper, valueLower = fused_pde(U[iElement], U[iNeighbors[iFaceIndex]], Sf[iFaceIndex], nus[iElement], gDiffs[iFaceIndex], zero(t), zero(t))
            @inbounds idx = ifelse(isOwner, ownerRelOwnerIdx[iFaceIndex], neighborRelNeighborIdx[iFaceIndex])
            @inbounds vals[idx] += ifelse(isOwner, valueLower, valueUpper)
            @inbounds diag += ifelse(!isOwner, valueLower, valueUpper)

        end
        @inbounds for iFace in numInteriors[iElement]+1:facesPerCell[iElement]
            @inbounds iFaceIndex = iFaces[startIndex+iFace]
            @inbounds bFaceIndex = bFaceMapping[iFaceIndex-numInternalFaces]
            if bFaceIndex != -1
                @inbounds diag, rhsx, rhsy, rhsz = fused_pde(bFaceValues[bFaceIndex], Sf[iFaceIndex], nus[iElement], gDiffs[iFaceIndex], diag, rhsx, rhsy, rhsz)
            end
        end
        @inbounds RHS[iElement] += rhsx
        @inbounds RHS[iElement+nCells] += rhsy
        @inbounds RHS[iElement+nCells+nCells] += rhsz
        @inbounds vals[rowOffsets[iElement]] += diag
    end
end


function getFaceBasedGpuInput2(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu_g = CuArray(input.nu)
    nCells::Int32 = length(mesh.cells)
    RHS = CUDA.zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = gpu_precalcOffsets(input)
    vals = CUDA.zeros(P, entriesNeeded)
    rows = CUDA.zeros(Int32, entriesNeeded)
    cols = CUDA.zeros(Int32, entriesNeeded)
    prepareRelativeIndices!(input)
    bFaceValues, U, bFaceMapping = gpu_getFaceValues(input)
    batches = group(x -> x.batchId, input.mesh.faces)
    gpuBatches = [facesToGPUarrays(b) for b in batches.values]
    return gpuBatches, offsets, nu_g, rows, cols, vals, U, bFaceValues, RHS, input.mesh.numInteriorFaces, bFaceMapping
end

function getBatchedFaceBasedGpuInput2(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    getGreedyEdgeColoring(input)
    gpuBatches, offsets, nus, rows, cols, vals, U, bFaceValues, RHS, N, bFaceMapping = getFaceBasedGpuInput2(input)
    return offsets, nus, rows, cols, vals, U, bFaceValues, RHS, bFaceMapping, gpuBatches
end

function run_batchedFace_abstract_presorted(
    offsets,
    nus,
    rows,
    cols,
    vals,
    U,
    bFaceValues,
    RHS,
    bFaceMapping,
    batches,
    fused_pde
)
    backend = CUDABackend()
    for color in 1:length(batches)-1
        kernel_FusedBatchedFaceAssembly_presorted(backend, 256)(
            batches[color]...,
            offsets,
            nus,
            rows,
            cols,
            vals,
            U,
            fused_pde;
            ndrange=length(batches[color][1])
        )
        KernelAbstractions.synchronize(backend)
    end

    kernel_abstract_boundaryFace_presorted(backend, 256)(
        batches[end][1],
        batches[end][3],
        offsets,
        nus,
        vals,
        bFaceValues,
        RHS,
        batches[end][6],
        bFaceMapping,
        fused_pde,
        ndrange=length(batches[end][1])
    )
    KernelAbstractions.synchronize(backend)

    return rows, cols, vals, RHS
end

function run_batchedFace_abstract_joinSmaller(
    offsets,
    nus,
    rows,
    cols,
    vals,
    U,
    bFaceValues,
    RHS,
    bFaceMapping,
    batches,
    joinedBatches,
    fused_pde
)
    backend = CUDABackend()
    for color in eachindex(batches)
        if color == joinedBatches
            break
        end
        kernel_FusedBatchedFaceAssembly_presorted(backend, 256)(
            batches[color]...,
            offsets,
            nus,
            rows,
            cols,
            vals,
            U,
            fused_pde;
            ndrange=length(batches[color][1])
        )
        KernelAbstractions.synchronize(backend)
    end
    for joined in joinedBatches:length(batches)-1
        kernel_joinedBatches(backend, 256)(
            batches[joined]...,
            offsets,
            nus,
            rows,
            cols,
            vals,
            U,
            fused_pde;
            ndrange=length(batches[joined][1])
        )
        # no need to sync here, its atomic anyways
    end
    KernelAbstractions.synchronize(backend)

    kernel_abstract_boundaryFace_presorted(backend, 256)(
        batches[end][1],
        batches[end][3],
        offsets,
        nus,
        vals,
        bFaceValues,
        RHS,
        batches[end][6],
        bFaceMapping,
        fused_pde,
        ndrange=length(batches[end][1])
    )
    KernelAbstractions.synchronize(backend)

    return rows, cols, vals, RHS
end

@kernel function kernel_FusedBatchedFaceAssembly_presorted(
    @Const(iOwners),
    @Const(iNeighbors),
    @Const(gDiffs),
    @Const(relativeToOwner),
    @Const(relativesN),
    @Const(Sf),
    @Const(offsets),
    @Const(nus),
    rows,
    cols,
    vals,
    @Const(U),
    @Const(fused_pde)
)
    t = typeof(nus[1])
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    upper, lower = zero(t), zero(t)
    upper, lower = fused_pde(U[iOwner], U[iNeighbor], Sf[iFace], nus[iOwner], gDiffs[iFace], upper, lower)

    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    vals[idx] += upper

    idx = offsets[iOwner] + relativeToOwner[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] += lower

    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    vals[idx] += lower

    idx = offsets[iNeighbor] + relativesN[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += upper
end


@kernel function kernel_joinedBatches(
    @Const(iOwners),
    @Const(iNeighbors),
    @Const(gDiffs),
    @Const(relativeToOwner),
    @Const(relativeToNeighbor),
    @Const(Sf),
    @Const(offsets),
    @Const(nus),
    rows,
    cols,
    vals,
    @Const(U),
    @Const(fused_pde)
)
    t = typeof(nus[1])
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    upper, lower = zero(t), zero(t)
    upper, lower = fused_pde(U[iOwner], U[iNeighbor], Sf[iFace], nus[iOwner], gDiffs[iFace], upper, lower)

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
end

@kernel function kernel_abstract_boundaryFace_presorted(
    @Const(iOwners),
    @Const(gDiffs),
    @Const(offsets),
    @Const(nus),
    vals,
    @Const(bFaceValues),
    RHS,
    @Const(Sf),
    @Const(bFaceMapping),
    @Const(fused_pde)
)
    t = typeof(nus[1])
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    bFaceIndex = bFaceMapping[iFace]
    if bFaceIndex != -1
        diag, rhsx, rhsy, rhsz = zero(t), zero(t), zero(t), zero(t)
        diag, rhsx, rhsy, rhsz = fused_pde(bFaceValues[bFaceIndex], Sf[iFace], nus[iOwner], gDiffs[iFace], diag, rhsx, rhsy, rhsz)

        # Diagonal Entry        
        idx = offsets[iOwner]
        Atomix.@atomic vals[idx] += diag

        # RHS/Source
        nCells = length(nus)
        Atomix.@atomic RHS[iOwner] += rhsx
        Atomix.@atomic RHS[iOwner+nCells] += rhsy
        Atomix.@atomic RHS[iOwner+nCells+nCells] += rhsz
    end
end