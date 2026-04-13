include("gpu_abstract.jl")

function JoinedEdgeColoring(input::MatrixAssemblyInput)
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
                face.batchId = ifelse(id >= 6, 6, id)
                faceColorMapping[face.index] = ifelse(id >= 6, 6, id)
                push!(usedColors, face.batchId)
                go = false
            end
        end
    end
    # return maximum(faceColorMapping), CuArray{Int32}(faceColorMapping)
    return maximum(faceColorMapping), faceColorMapping
end

function DumbEdgeColoring(input::MatrixAssemblyInput)
    mesh = input.mesh
    parts = Iterators.partition(1:mesh.numInteriorFaces, cld(mesh.numInteriorFaces, 2))
    for (i, part) in enumerate(parts)
        for p in part
            mesh.faces[p].batchId = i
        end
    end
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

function getBatchedFaceBasedGpuInput2(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
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


function faceLikeInput(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    colors, _ = getGreedyEdgeColoring(input)
    grouped = group(x -> x.iNeighbor < 1, input.mesh.faces)
    gpubatches = [toGPUSOAs(VectorToSOAs(bx)) for bx in grouped.values]
    bFaceValues, U, bFaceMapping = gpu_getFaceValues(input)
    nus = CuArray(input.nu)
    entriesNeeded::Int32 = length(input.mesh.cells) + 2 * input.mesh.numInteriorFaces
    RHS = CUDA.zeros(P, length(input.mesh.cells) * 3)
    vals = CUDA.zeros(P, entriesNeeded)
    return gpubatches, U, nus, bFaceValues, bFaceMapping, vals, RHS, colors
end

function FusedBatchedAssembly(batches, U, nus, bFaceValues, bFaceMapping, vals, RHS, colors, fused_pde, wg, nd)
    backend = CUDABackend()
    internalKernel! = nongrouped_internal_nostride(backend, wg)
    batched_boundary(backend, wg)(
        batches[2].iOwner,
        batches[2].gDiff,
        batches[2].ownerIdx,
        batches[2].Sf,
        nus,
        bFaceValues,
        bFaceMapping,
        fused_pde,
        vals,
        RHS;
        ndrange=nd
    )
    for color in 1:colors
        internalKernel!(
            batches[1].iOwner,
            batches[1].iNeighbor,
            batches[1].gDiff,
            batches[1].ownerIdx,
            batches[1].ownerRelOwnerIdx,
            batches[1].neighborIdx,
            batches[1].neighborRelNeighborIdx,
            batches[1].Sf,
            batches[1].batchId,
            color,
            nus,
            U,
            fused_pde,
            vals;
            ndrange=nd
        )
        KernelAbstractions.synchronize(backend)
    end
    return vals, RHS

end

function FusedBatchedAssembly2(batches, U, nus, bFaceValues, bFaceMapping, vals, RHS, colors, fused_pde, wg, nd)
    backend = CUDABackend()
    internalKernel! = nongrouped_internal(backend, wg)
    for color in 1:colors
        internalKernel!(
            batches[color].iOwner,
            batches[color].iNeighbor,
            batches[color].gDiff,
            batches[color].ownerIdx,
            batches[color].ownerRelOwnerIdx,
            batches[color].neighborIdx,
            batches[color].neighborRelNeighborIdx,
            batches[color].Sf,
            batches[color].batchId,
            color,
            nus,
            U,
            fused_pde,
            vals;
            ndrange=nd
        )
        KernelAbstractions.synchronize(backend)
    end
    batched_boundary(backend, wg)(
        batches[end].iOwner,
        batches[end].gDiff,
        batches[end].ownerIdx,
        batches[end].Sf,
        nus,
        bFaceValues,
        bFaceMapping,
        fused_pde,
        vals,
        RHS;
        ndrange=nd
    )
    KernelAbstractions.synchronize(backend)
    return vals, RHS
end


@kernel function nongrouped_internal(
    @Const(iOwners),
    @Const(iNeighbors),
    @Const(gDiffs),
    @Const(ownerIdx),
    @Const(ownerRelOwnerIdx),
    @Const(neighborIdx),
    @Const(neighborRelNeighborIdx),
    @Const(Sf),
    @Const(colors),
    @Const(currentColor),
    @Const(nus),
    @Const(U),
    @Const(fused_pde),
    vals,
)
    t = eltype(nus)
    nFaces = length(iOwners)
    @inbounds stride = @ndrange()[1]
    i = @index(Global, Linear)
    iFace = i
    while iFace <= nFaces
        @inbounds if currentColor != colors[iFace]
            iFace += stride
            continue
        end
        @inbounds iOwner = iOwners[iFace]
        @inbounds iNeighbor = iNeighbors[iFace]

        upper, lower = fused_pde(U[iOwner], U[iNeighbor], Sf[iFace], nus[iOwner], gDiffs[iFace], zero(t), zero(t))
        @inbounds vals[ownerIdx[iFace]] += upper
        @inbounds vals[ownerRelOwnerIdx[iFace]] += lower
        @inbounds vals[neighborIdx[iFace]] += lower
        @inbounds vals[neighborRelNeighborIdx[iFace]] += upper
        iFace += stride
    end
end

@kernel function nongrouped_internal_nostride(
    @Const(iOwners),
    @Const(iNeighbors),
    @Const(gDiffs),
    @Const(ownerIdx),
    @Const(ownerRelOwnerIdx),
    @Const(neighborIdx),
    @Const(neighborRelNeighborIdx),
    @Const(Sf),
    @Const(colors),
    @Const(currentColor),
    @Const(nus),
    @Const(U),
    @Const(fused_pde),
    vals,
)
    t = eltype(nus)
    nFaces = length(iOwners)
    @inbounds stride = @ndrange()[1]
    iFace = @index(Global)
    if iFace <= nFaces
        @inbounds if currentColor == colors[iFace]
            @inbounds iOwner = iOwners[iFace]
            @inbounds iNeighbor = iNeighbors[iFace]
            
            @inbounds upper, lower = fused_pde(U[iOwner], U[iNeighbor], Sf[iFace], nus[iOwner], gDiffs[iFace], zero(t), zero(t))
            @inbounds vals[ownerIdx[iFace]] += upper
            @inbounds vals[ownerRelOwnerIdx[iFace]] += lower
            @inbounds vals[neighborIdx[iFace]] += lower
            @inbounds vals[neighborRelNeighborIdx[iFace]] += upper
        end
    end
end

@kernel function batched_boundary(
    @Const(iOwners),
    @Const(gDiffs),
    @Const(ownerIdx),
    @Const(Sf),
    @Const(nus),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(fused_pde),
    vals,
    RHS
)
    t = eltype(nus)
    stride = @ndrange()[1]
    i = @index(Global, Linear)
    iFace = i
    while iFace <= length(bFaceMapping)
        bFaceIndex = bFaceMapping[iFace]
        if bFaceIndex != -1
            iOwner = iOwners[iFace]
            diag, rhsx, rhsy, rhsz = zero(t), zero(t), zero(t), zero(t)
            diag, rhsx, rhsy, rhsz = fused_pde(bFaceValues[bFaceIndex], Sf[iFace], nus[iOwner], gDiffs[iFace], diag, rhsx, rhsy, rhsz)

            # Diagonal Entry        
            Atomix.@atomic vals[ownerIdx[iFace]] += diag

            # RHS/Source
            nCells = length(nus)
            Atomix.@atomic RHS[iOwner] += rhsx
            Atomix.@atomic RHS[iOwner+nCells] += rhsy
            Atomix.@atomic RHS[iOwner+nCells+nCells] += rhsz
        end
        iFace += stride
    end
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

function GlobalFaceAssembly(batches, U, nus, bFaceValues, bFaceMapping, vals, RHS, fused_pde, wg, nd)
    backend = CUDABackend()
    GlobalFaceBasedKernel(backend, wg)(
        batches.iOwner,
        batches.iNeighbor,
        batches.gDiff,
        batches.ownerIdx,
        batches.ownerRelOwnerIdx,
        batches.neighborIdx,
        batches.neighborRelNeighborIdx,
        batches.Sf,
        nus,
        U,
        bFaceValues,
        bFaceMapping,
        fused_pde,
        vals,
        RHS;
        ndrange=nd
    )
    KernelAbstractions.synchronize(backend)
    return vals, RHS
end


@kernel function GlobalFaceBasedKernel(
    @Const(iOwners),
    @Const(iNeighbors),
    @Const(gDiffs),
    @Const(ownerIdx),
    @Const(ownerRelOwnerIdx),
    @Const(neighborIdx),
    @Const(neighborRelNeighborIdx),
    @Const(Sf),
    @Const(nus),
    @Const(U),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(fused_pde),
    vals,
    RHS
)
    t = eltype(nus)
    numFaces = length(iOwners)
    numInternalFaces = numFaces - length(bFaceMapping)
    iFace = @index(Global)
    @inbounds iOwner = iOwners[iFace]
    @inbounds iNeighbor = iNeighbors[iFace]
    nCells = length(nus)
    if iNeighbor > 0
        @inbounds upper, lower = fused_pde(U[iOwner], U[iNeighbor], Sf[iFace], nus[iOwner], gDiffs[iFace], zero(t), zero(t))

        @inbounds Atomix.@atomic vals[ownerIdx[iFace]] += upper
        @inbounds vals[ownerRelOwnerIdx[iFace]] += lower
        @inbounds Atomix.@atomic vals[neighborIdx[iFace]] += lower
        @inbounds vals[neighborRelNeighborIdx[iFace]] += upper
    else
        @inbounds relativeFaceIndex = iFace - numInternalFaces
        @inbounds bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            @inbounds diag, rhsx, rhsy, rhsz = fused_pde(bFaceValues[bFaceIndex], Sf[iFace], nus[iOwner], gDiffs[iFace], zero(t), zero(t), zero(t), zero(t))
            # Diagonal Entry     
            @inbounds Atomix.@atomic vals[ownerIdx[iFace]] += diag

            # RHS/Source
            @inbounds Atomix.@atomic RHS[iOwner] += rhsx
            @inbounds Atomix.@atomic RHS[iOwner+nCells] += rhsy
            @inbounds Atomix.@atomic RHS[iOwner+nCells+nCells] += rhsz
        end
    end
end


function FaceBasedAssembly(batches, U, nus, bFaceValues, bFaceMapping, vals, RHS, fused_pde, wg, nd)
    backend = CUDABackend()
    FaceBasedInternalKernel(backend, wg)(
        batches[1].iOwner,
        batches[1].iNeighbor,
        batches[1].gDiff,
        batches[1].ownerIdx,
        batches[1].ownerRelOwnerIdx,
        batches[1].neighborIdx,
        batches[1].neighborRelNeighborIdx,
        batches[1].Sf,
        nus,
        U,
        fused_pde,
        vals;
        ndrange=nd
    )
    FaceBasedBoundaryKernel(backend, wg)(
        batches[2].iOwner,
        batches[2].gDiff,
        batches[2].ownerIdx,
        batches[2].Sf,
        nus,
        bFaceValues,
        bFaceMapping,
        fused_pde,
        vals,
        RHS;
        ndrange=nd
    )
    KernelAbstractions.synchronize(backend)
    return vals, RHS

end


@kernel function FaceBasedInternalKernel(
    @Const(iOwners),
    @Const(iNeighbors),
    @Const(gDiffs),
    @Const(ownerIdx),
    @Const(ownerRelOwnerIdx),
    @Const(neighborIdx),
    @Const(neighborRelNeighborIdx),
    @Const(Sf),
    @Const(nus),
    @Const(U),
    @Const(fused_pde),
    vals,
)
    t = eltype(nus)
    nFaces = length(iOwners)
    @inbounds stride = @ndrange()[1]
    iFace = @index(Global)
    if iFace <= nFaces
        @inbounds iOwner = iOwners[iFace]
        @inbounds iNeighbor = iNeighbors[iFace]
        
        @inbounds upper, lower = fused_pde(U[iOwner], U[iNeighbor], Sf[iFace], nus[iOwner], gDiffs[iFace], zero(t), zero(t))
        @inbounds Atomix.@atomic vals[ownerIdx[iFace]] += upper
        @inbounds vals[ownerRelOwnerIdx[iFace]] += lower
        @inbounds Atomix.@atomic vals[neighborIdx[iFace]] += lower
        @inbounds vals[neighborRelNeighborIdx[iFace]] += upper
    end
end

@kernel function FaceBasedBoundaryKernel(
    @Const(iOwners),
    @Const(gDiffs),
    @Const(ownerIdx),
    @Const(Sf),
    @Const(nus),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(fused_pde),
    vals,
    RHS
)
    t = eltype(nus)
    iFace = @index(Global)
    if iFace <= length(bFaceMapping)
        @inbounds bFaceIndex = bFaceMapping[iFace]
        if bFaceIndex != -1
            @inbounds iOwner = iOwners[iFace]
            @inbounds diag, rhsx, rhsy, rhsz = fused_pde(bFaceValues[bFaceIndex], Sf[iFace], nus[iOwner], gDiffs[iFace], zero(t), zero(t), zero(t), zero(t))

            # Diagonal Entry        
            @inbounds Atomix.@atomic vals[ownerIdx[iFace]] += diag

            # RHS/Source
            nCells = length(nus)
            @inbounds Atomix.@atomic RHS[iOwner] += rhsx
            @inbounds Atomix.@atomic RHS[iOwner+nCells] += rhsy
            @inbounds Atomix.@atomic RHS[iOwner+nCells+nCells] += rhsz
        end
    end
end