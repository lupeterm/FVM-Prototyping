include("init.jl")
include("gpu_faceBased.jl")
include("gpu_helper.jl")

@kernel function batched_boundary_divlap(
    @Const(iOwners),
    @Const(gDiffs),
    @Const(ownerIdx),
    @Const(Sf),
    @Const(nus),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    vals,
    RHS,
)
    t = eltype(nus)
    iFace = @index(Global)
    if iFace <= length(bFaceMapping)
        bFaceIndex = bFaceMapping[iFace]
        if bFaceIndex != -1
            iOwner = iOwners[iFace]
            
            convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])
            diffusion = nus[iOwner] * gDiffs[iFace]
            Atomix.@atomic vals[ownerIdx[iFace]] += diffusion

            # RHS/Source
            nCells = length(nus)
            Atomix.@atomic RHS[iOwner] -= convection[1] - diffusion
            Atomix.@atomic RHS[iOwner+nCells] -= convection[2] - diffusion
            Atomix.@atomic RHS[iOwner+nCells+nCells] -= convection[3] - diffusion
        end
    end
end

@kernel function batched_boundary_divOnly(
    @Const(iOwners),
    @Const(gDiffs),
    @Const(ownerIdx),
    @Const(Sf),
    @Const(nus),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    vals,
    RHS
)
    t = eltype(nus)
    iFace = @index(Global)
    if iFace <= length(bFaceMapping)
        bFaceIndex = bFaceMapping[iFace]
        if bFaceIndex != -1
            iOwner = iOwners[iFace]
            
            convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])

            # RHS/Source
            nCells = length(nus)
            Atomix.@atomic RHS[iOwner] -= convection[1]
            Atomix.@atomic RHS[iOwner+nCells] -= convection[2]
            Atomix.@atomic RHS[iOwner+nCells+nCells] -= convection[3]
        end
    end
end

@kernel function batched_boundary_laplaceOnly(
    @Const(iOwners),
    @Const(gDiffs),
    @Const(ownerIdx),
    @Const(Sf),
    @Const(nus),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    vals,
    RHS
)
    t = eltype(nus)
    iFace = @index(Global)
    if iFace <= length(bFaceMapping)
        bFaceIndex = bFaceMapping[iFace]
        if bFaceIndex != -1
            iOwner = iOwners[iFace]
            
            diffusion = nus[iOwner] * gDiffs[iFace]
            # Diagonal Entry        
            Atomix.@atomic vals[ownerIdx[iFace]] += diffusion

            # RHS/Source
            nCells = length(nus)
            Atomix.@atomic RHS[iOwner] -= diffusion
            Atomix.@atomic RHS[iOwner+nCells] -= diffusion
            Atomix.@atomic RHS[iOwner+nCells+nCells] -= diffusion
        end
    end
end


function gpu_LaplaceOnlyBatchedFaceAssemblyRunner(batches, U, nus, bFaceValues, bFaceMapping, vals, RHS)
    backend = CUDABackend()
    internalKernel! = kernel_LaplaceOnlyBatchedFaceAssembly(backend, 64)
    for id in eachindex(batches)
        if id == length(batches)
            break
        end
        internalKernel!(
            batches[id].iOwner,
            batches[id].iNeighbor,
            batches[id].gDiff,
            batches[id].ownerIdx,
            batches[id].ownerRelOwnerIdx,
            batches[id].neighborIdx,
            batches[id].neighborRelNeighborIdx,
            batches[id].Sf,
            batches[id].batchId,
            id,
            nus,
            U,
            vals;
            ndrange=16384
        )
        KernelAbstractions.synchronize(backend)
    end
    batched_boundary_laplaceOnly(backend, 64)(
        batches[end].iOwner,
        batches[end].gDiff,
        batches[end].ownerIdx,
        batches[end].Sf,
        nus,
        bFaceValues,
        bFaceMapping,
        vals,
        RHS;
        ndrange=16384
    )
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_LaplaceOnlyBatchedFaceAssembly(
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
    vals,
)
    t = eltype(nus)
    nFaces = length(iOwners)
    stride = @ndrange()[1]
    i = @index(Global, Linear)
    iFace = i
    while iFace <= nFaces
        if currentColor != colors[iFace]
            iFace += stride
            continue
        end
        iOwner = iOwners[iFace]
        iNeighbor = iNeighbors[iFace]

        # Diffusion
        diffusion = nus[iOwner] * gDiffs[iFace]
        vals[ownerIdx[iFace]] += diffusion
        vals[ownerRelOwnerIdx[iFace]] -= diffusion
        vals[neighborIdx[iFace]] -= diffusion
        vals[neighborRelNeighborIdx[iFace]] += diffusion
        iFace += stride
    end
end

function gpu_DivOnlyPrecalculatedWeightsBatchedFaceAssemblyRunner(batches, U, nus, bFaceValues, bFaceMapping, vals, RHS, weights)
    backend = CUDABackend()
    internalKernel! = kernel_DivOnlyPrecalculatedWeightsBatchedFaceAssembly(backend, 64)
    for id in eachindex(batches)
        if id == length(batches)
            continue
        end
        internalKernel!(
            batches[id].iOwner,
            batches[id].iNeighbor,
            batches[id].gDiff,
            batches[id].ownerIdx,
            batches[id].ownerRelOwnerIdx,
            batches[id].neighborIdx,
            batches[id].neighborRelNeighborIdx,
            batches[id].Sf,
            batches[id].batchId,
            id,
            nus,
            U, vals,
            weights;
            ndrange=16384
        )
        KernelAbstractions.synchronize(backend)
    end
    batched_boundary_divOnly(backend, 64)(
        batches[end].iOwner,
        batches[end].gDiff,
        batches[end].ownerIdx,
        batches[end].Sf,
        nus,
        bFaceValues,
        bFaceMapping, vals,
        RHS;
        ndrange=16384
    )
    KernelAbstractions.synchronize(backend)
end


@kernel function kernel_DivOnlyPrecalculatedWeightsBatchedFaceAssembly(
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
    vals,
    weights
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

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = weights[iFace]                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        upper = ϕf * weights_f
        lower = -ϕf * (1 - weights_f)

        vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
        iFace += stride
    end
end



function gpu_DivOnlyHardcodedDivBatchedFaceAssemblyRunner(batches, U, nus, bFaceValues, bFaceMapping, vals, RHS, upwindOrCdf)
    backend = CUDABackend()
    internalKernel! = ifelse(upwindOrCdf == "CDF", kernel_DivOnlyHardcodedDivCDFBatchedFaceAssembly(backend, 64), kernel_DivOnlyHardcodedDivUpwindBatchedFaceAssembly(backend, 64))
    for id in eachindex(batches)
        if id == length(batches)
            continue
        end
        internalKernel!(
            batches[id].iOwner,
            batches[id].iNeighbor,
            batches[id].gDiff,
            batches[id].ownerIdx,
            batches[id].ownerRelOwnerIdx,
            batches[id].neighborIdx,
            batches[id].neighborRelNeighborIdx,
            batches[id].Sf,
            batches[id].batchId,
            id,
            nus,
            U, vals;
            ndrange=16384
        )
        KernelAbstractions.synchronize(backend)
    end
    batched_boundary_divOnly(backend, 64)(
        batches[end].iOwner,
        batches[end].gDiff,
        batches[end].ownerIdx,
        batches[end].Sf,
        nus,
        bFaceValues,
        bFaceMapping, vals,
        RHS;
        ndrange=16384
    )
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_DivOnlyHardcodedDivCDFBatchedFaceAssembly(
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

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = centralDifferencing(ϕf)                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        upper = ϕf * weights_f
        lower = -ϕf * (1 - weights_f)

        vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
        iFace += stride
    end
end

@kernel function kernel_DivOnlyHardcodedDivUpwindBatchedFaceAssembly(
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

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = upwind(ϕf)                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        upper = ϕf * weights_f
        lower = -ϕf * (1 - weights_f)

        vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
        iFace += stride
    end
end


function gpu_DivOnlyDynamicBatchedFaceAssemblyRunner(batches, U, nus, bFaceValues, bFaceMapping, vals, RHS, div)
    backend = CUDABackend()
    internalKernel! = kernel_DivOnlyDynamicBatchedFaceAssembly(backend, 64)
    for id in eachindex(batches)
        if id == length(batches)
            continue
        end
        internalKernel!(
            batches[id].iOwner,
            batches[id].iNeighbor,
            batches[id].gDiff,
            batches[id].ownerIdx,
            batches[id].ownerRelOwnerIdx,
            batches[id].neighborIdx,
            batches[id].neighborRelNeighborIdx,
            batches[id].Sf,
            batches[id].batchId,
            id,
            nus,
            U, vals,
            div;
            ndrange=16384
        )
        KernelAbstractions.synchronize(backend)
    end
    batched_boundary_divOnly(backend, 64)(
        batches[end].iOwner,
        batches[end].gDiff,
        batches[end].ownerIdx,
        batches[end].Sf,
        nus,
        bFaceValues,
        bFaceMapping, vals,
        RHS;
        ndrange=16384
    )
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_DivOnlyDynamicBatchedFaceAssembly(
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
    vals,
    div
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

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = div(ϕf)                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        upper = ϕf * weights_f
        lower = -ϕf * (1 - weights_f)

        vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
        iFace += stride
    end
end

function gpu_PrecalculatedWeightsBatchedFaceAssemblyRunner(batches, U, nus, bFaceValues, bFaceMapping, vals, RHS, weights)
    backend = CUDABackend()
    internalKernel! = kernel_PrecalculatedWeightsBatchedFaceAssembly(backend, 64)
    for color in eachindex(batches)
        if color == length(batches)
            continue
        end
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
            vals,
            weights;
            ndrange=16384
        )
        KernelAbstractions.synchronize(backend)
    end
    batched_boundary_divlap(backend, 64)(
        batches[2].iOwner,
        batches[2].gDiff,
        batches[2].ownerIdx,
        batches[2].Sf,
        nus,
        bFaceValues,
        bFaceMapping,
        vals,
        RHS;
        ndrange=16384
    )
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_PrecalculatedWeightsBatchedFaceAssembly(
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
    vals,
    weights
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

        # Diffusion
        diffusion = nus[iOwner] * gDiffs[iFace]

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = weights[iFace]                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        upper = ϕf * weights_f + diffusion
        lower = -ϕf * (1 - weights_f) - diffusion

        vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
        iFace += stride
    end
end


function gpu_HardcodedBatchedFaceAssemblyRunner(batches, U, nus, bFaceValues, bFaceMapping, vals, RHS, upwindOrCdf)
    backend = CUDABackend()
    internalKernel! = ifelse(upwindOrCdf == "CDF", kernel_HardcodedCDFBatchedFaceAssembly(backend, 64), kernel_HardcodedUpwindBatchedFaceAssembly(backend, 64))
    for color in eachindex(batches)
        if color == length(batches)
            continue
        end
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
            U, vals;
            ndrange=16384
        )
        KernelAbstractions.synchronize(backend)
    end
    batched_boundary_divlap(backend, 64)(
        batches[2].iOwner,
        batches[2].gDiff,
        batches[2].ownerIdx,
        batches[2].Sf,
        nus,
        bFaceValues,
        bFaceMapping, vals,
        RHS;
        ndrange=16384
    )
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_HardcodedCDFBatchedFaceAssembly(
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

        # Diffusion
        diffusion = nus[iOwner] * gDiffs[iFace]

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = centralDifferencing(ϕf)                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        upper = ϕf * weights_f + diffusion
        lower = -ϕf * (1 - weights_f) - diffusion

        vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
        iFace += stride
    end
end


@kernel function kernel_HardcodedUpwindBatchedFaceAssembly(
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
    vals
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

        # Diffusion
        diffusion = nus[iOwner] * gDiffs[iFace]

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                             # flux through the face
        weights_f = upwind(ϕf)                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        upper = ϕf * weights_f + diffusion
        lower = -ϕf * (1 - weights_f) - diffusion

        vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
        iFace += stride
    end
end



function gpu_DynamicBatchedFaceAssemblyRunner(batches, U, nus, bFaceValues, bFaceMapping, vals, RHS, div)
    backend = CUDABackend()
    internalKernel! = kernel_DynamicBatchedFaceAssembly(backend, 64)
    for color in eachindex(batches)
        if color == length(batches)
            continue
        end
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
            U, vals,
            div;
            ndrange=16384
        )
        KernelAbstractions.synchronize(backend)
    end
    batched_boundary_divlap(backend, 64)(
        batches[2].iOwner,
        batches[2].gDiff,
        batches[2].ownerIdx,
        batches[2].Sf,
        nus,
        bFaceValues,
        bFaceMapping, vals,
        RHS;
        ndrange=16384
    )
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_DynamicBatchedFaceAssembly(
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
    vals,
    div
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

        # Diffusion
        diffusion = nus[iOwner] * gDiffs[iFace]

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = div(ϕf)                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        upper = ϕf * weights_f + diffusion
        lower = -ϕf * (1 - weights_f) - diffusion

        vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
        iFace += stride
    end
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