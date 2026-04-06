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
    internalKernel! = nongrouped_internal(backend, wg)
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
    KernelAbstractions.synchronize(backend)
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


@kernel function nongrouped_internal2(
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
    stride = @ndrange()[1]
    i = @index(Global, Linear)
    iFace = i
    while iFace < nFaces
        if currentColor != colors[iFace]
            iFace += stride
            continue
        end
        iOwner = iOwners[iFace]
        iNeighbor = iNeighbors[iFace]

        upper, lower = fused_pde(U[iOwner], U[iNeighbor], Sf[iFace], nus[iOwner], gDiffs[iFace], zero(t), zero(t))
        vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
        iFace += stride
    end
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
    stride = @ndrange()[1]
    i = @index(Global, Linear)
    iFace = i
    while iFace < nFaces
        if currentColor != colors[iFace]
            iFace += stride
            continue
        end
        iOwner = iOwners[iFace]
        iNeighbor = iNeighbors[iFace]

        upper, lower = fused_pde(U[iOwner], U[iNeighbor], Sf[iFace], nus[iOwner], gDiffs[iFace], zero(t), zero(t))
        vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
        iFace += stride
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
    iFace = @index(Global)
    if iFace <= length(bFaceMapping)
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

function FaceAssembly(batches, U, nus, bFaceValues, bFaceMapping, vals, RHS, fused_pde, wg, nd)
    backend = CUDABackend()
    # nthreads = cld(length(batches.iOwner), chunkSize)

    stridedfacekernel(backend, wg)(
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
end


@kernel function noStridefacekernel(
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
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    if iNeighbor > 0
        upper, lower = fused_pde(U[iOwner], U[iNeighbor], Sf[iFace], nus[iOwner], gDiffs[iFace], zero(t), zero(t))

        Atomix.@atomic vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        Atomix.@atomic vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
    else
        relativeFaceIndex = iFace - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            diag, rhsx, rhsy, rhsz = fused_pde(bFaceValues[bFaceIndex], Sf[iFace], nus[iOwner], gDiffs[iFace], zero(t), zero(t), zero(t), zero(t))
            # Diagonal Entry     
            Atomix.@atomic vals[ownerIdx[iFace]] += diag

            # RHS/Source
            nCells = length(nus)
            CUDA.@atomic RHS[iOwner] += rhsx
            CUDA.@atomic RHS[iOwner+nCells] += rhsy
            CUDA.@atomic RHS[iOwner+nCells+nCells] += rhsz
        end
    end
end

@kernel function stridedfacekernel(
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
    tid = @index(Global, Linear)
    stride = @ndrange()[1]
    iFace = tid
    while iFace < numFaces
        iOwner = iOwners[iFace]
        iNeighbor = iNeighbors[iFace]
        if iNeighbor > 0
            upper, lower = fused_pde(U[iOwner], U[iNeighbor], Sf[iFace], nus[iOwner], gDiffs[iFace], zero(t), zero(t))

            Atomix.@atomic vals[ownerIdx[iFace]] += upper
            vals[ownerRelOwnerIdx[iFace]] += lower
            Atomix.@atomic vals[neighborIdx[iFace]] += lower
            vals[neighborRelNeighborIdx[iFace]] += upper
        else
            relativeFaceIndex = iFace - numInternalFaces
            bFaceIndex = bFaceMapping[relativeFaceIndex]
            if bFaceIndex != -1
                diag, rhsx, rhsy, rhsz = fused_pde(bFaceValues[bFaceIndex], Sf[iFace], nus[iOwner], gDiffs[iFace], zero(t), zero(t), zero(t), zero(t))
                # Diagonal Entry     
                Atomix.@atomic vals[ownerIdx[iFace]] += diag

                # RHS/Source
                nCells = length(nus)
                CUDA.@atomic RHS[iOwner] += rhsx
                CUDA.@atomic RHS[iOwner+nCells] += rhsy
                CUDA.@atomic RHS[iOwner+nCells+nCells] += rhsz
            end
        end
        iFace += stride
    end
end

function test(args...)
    workgroups = (1,2,4,8,16,32, 64, 128, 256, 512, 1024)
    # 64, 128, 256, 512, 1024)
    multipliers = (32, 64, 128, 256)
    for wg in workgroups
        for m in multipliers
            nd = wg * m
            println("wg: $wg, nd: $nd")
            @time begin
                BatchedAssembly(args..., wg, nd)
            end
        end
    end
end


@kernel function empty_kernel!(
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
    i = @index(Global, Linear)
end

function b(batches, U, nus, bFaceValues, bFaceMapping, vals, RHS, fused_pde)
    backend = CUDABackend()

    # warmup: compile once
    KernelAbstractions.synchronize(backend)
    @btime begin
        for i in 1:6
            $empty_kernel!($backend, 256)(
                $batches.iOwner,
                $batches.iNeighbor,
                $batches.gDiff,
                $batches.ownerIdx,
                $batches.ownerRelOwnerIdx,
                $batches.neighborIdx,
                $batches.neighborRelNeighborIdx,
                $batches.Sf,
                $nus,
                $U,
                $bFaceValues,
                $bFaceMapping,
                $fused_pde,
                $vals,
                $RHS;
                ndrange=256
            )
            KernelAbstractions.synchronize($backend)
        end
    end
end