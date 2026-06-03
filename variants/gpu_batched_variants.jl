include("common.jl")

function FusedBatchedAssembly(batches, U, nus, bFaceValues, bFaceMapping, vals, RHS, colors, fused_pde)
    backend = CUDABackend()
    internalKernel! = fused_internal(backend, 64)
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
            vals
            ;
            ndrange=length(batches[1].batchId)
            )
            KernelAbstractions.synchronize(backend)
        end
    batched_boundary_fused(backend, 64)(
        batches[2].iOwner,
        batches[2].gDiff,
        batches[2].ownerIdx,
        batches[2].Sf,
        nus,
        bFaceValues,
        bFaceMapping,
        fused_pde,
        vals,
        RHS,
        length(batches[1].batchId);
        ndrange=length(bFaceValues)
    )
    return vals, RHS
end


@kernel function fused_internal(
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
    iFace = @index(Global)
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


@kernel function batched_boundary_fused(
    @Const(iOwners),
    @Const(gDiffs),
    @Const(ownerIdx),
    @Const(Sf),
    @Const(nus),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(fused_pde),
    vals,
    RHS,
    nInternalFaces
)
    t = eltype(nus)
    iFace = @index(Global)
    bFaceIndex = bFaceMapping[iFace]
    if bFaceIndex != -1
        iOwner = iOwners[iFace]
        diag, rhsx, rhsy, rhsz = fused_pde(bFaceValues[bFaceIndex], Sf[iFace], nus[iOwner], gDiffs[iFace], zero(t), zero(t), zero(t), zero(t))

        # Diagonal Entry        
        Atomix.@atomic vals[ownerIdx[iFace]] += diag

        # RHS/Source
        nCells = length(nus)
        Atomix.@atomic RHS[iOwner] += rhsx
        Atomix.@atomic RHS[iOwner+nCells] += rhsy
        Atomix.@atomic RHS[iOwner+nCells+nCells] += rhsz
    end
end




function PrecalculatedWeightsBatchedAssembly(batches, U, nus, bFaceValues, bFaceMapping, vals, RHS, colors, weights)
    backend = CUDABackend()
    internalKernel! = precalc_internal(backend, 64)
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
            weights,
            vals;
            ndrange=length(batches[1].batchId)
            )
            KernelAbstractions.synchronize(backend)
        end
    batched_boundary_nonfused(backend, 64)(
        batches[2].iOwner,
        batches[2].gDiff,
        batches[2].ownerIdx,
        batches[2].Sf,
        nus,
        bFaceValues,
        bFaceMapping,
        vals,
                RHS,
        length(batches[1].batchId);
        ndrange=length(bFaceValues)
    )
    return vals, RHS
end


@kernel function precalc_internal(
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
    @Const(weights),
    vals,
)
    iFace = @index(Global)
    @inbounds if currentColor == colors[iFace]
        @inbounds iOwner = iOwners[iFace]
        @inbounds iNeighbor = iNeighbors[iFace]
        
        diffusion = nus[iOwner] * gDiffs[iFace]

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                      # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                                 # flux through the face
        weights_f = weights[iFace]                              # get weight of transport variable interpolation 
        upper = ϕf * weights_f + diffusion
        lower = -ϕf * (1 - weights_f) - diffusion

        @inbounds vals[ownerIdx[iFace]] += upper
        @inbounds vals[ownerRelOwnerIdx[iFace]] += lower
        @inbounds vals[neighborIdx[iFace]] += lower
        @inbounds vals[neighborRelNeighborIdx[iFace]] += upper
    end
end


@kernel function batched_boundary_nonfused(
    @Const(iOwners),
    @Const(gDiffs),
    @Const(ownerIdx),
    @Const(Sf),
    @Const(nus),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    vals,
    RHS,
    nInternalFaces
)
    t = eltype(nus)
    nCells = length(nus)
    iFace = @index(Global)
    bFaceIndex = bFaceMapping[iFace]
    if bFaceIndex != -1
        iOwner = iOwners[iFace]
        convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])
        diffusion = nus[iOwner] * gDiffs[iFace]
        Atomix.@atomic vals[ownerIdx[iFace]] -= diffusion    # x

        # RHS/Source
        Atomix.@atomic RHS[iOwner] -= convection[1] - diffusion
        Atomix.@atomic RHS[iOwner+nCells] -= convection[2] - diffusion
        Atomix.@atomic RHS[iOwner+nCells+nCells] -= convection[3] - diffusion
    end
end



function DynamicBatchedAssembly(batches, U, nus, bFaceValues, bFaceMapping, vals, RHS, colors, func)
    backend = CUDABackend()
    internalKernel! = dynamic_internal(backend, 64)
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
            func,
            vals;
            ndrange=length(batches[1].batchId)
            )
            KernelAbstractions.synchronize(backend)
        end
    batched_boundary_nonfused(backend, 64)(
        batches[2].iOwner,
        batches[2].gDiff,
        batches[2].ownerIdx,
        batches[2].Sf,
        nus,
        bFaceValues,
        bFaceMapping,
        vals,
                RHS,
        length(batches[1].batchId);
        ndrange=length(bFaceValues)
    )
    return vals, RHS
end


@kernel function dynamic_internal(
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
    @Const(func),
    vals,
)
    iFace = @index(Global)
    @inbounds if currentColor == colors[iFace]
        @inbounds iOwner = iOwners[iFace]
        @inbounds iNeighbor = iNeighbors[iFace]
        
        diffusion = nus[iOwner] * gDiffs[iFace]

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                      # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                                 # flux through the face
        weights_f = func(ϕf)                              # get weight of transport variable interpolation 
        upper = ϕf * weights_f + diffusion
        lower = -ϕf * (1 - weights_f) - diffusion

        @inbounds vals[ownerIdx[iFace]] += upper
        @inbounds vals[ownerRelOwnerIdx[iFace]] += lower
        @inbounds vals[neighborIdx[iFace]] += lower
        @inbounds vals[neighborRelNeighborIdx[iFace]] += upper
    end
end


function HardcodedBatchedAssembly(batches, U, nus, bFaceValues, bFaceMapping, vals, RHS, colors, whichone)
    backend = CUDABackend()
    internalKernel! = ifelse(whichone =="CDF", cdf_internal(backend, 64), upwind_internal(backend,64))
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
            vals;
            ndrange=length(batches[1].batchId)
            )
            KernelAbstractions.synchronize(backend)
        end
    batched_boundary_nonfused(backend, 64)(
        batches[2].iOwner,
        batches[2].gDiff,
        batches[2].ownerIdx,
        batches[2].Sf,
        nus,
        bFaceValues,
        bFaceMapping,
        vals,
        RHS,
        length(batches[1].batchId);
        ndrange=length(bFaceValues)
    )
    return vals, RHS
end


@kernel function cdf_internal(
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
    iFace = @index(Global)
    if currentColor == colors[iFace]
        iOwner = iOwners[iFace]
        iNeighbor = iNeighbors[iFace]
        
        diffusion = nus[iOwner] * gDiffs[iFace]

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                      # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                                 # flux through the face
        weights_f = cdf_f(ϕf)                              # get weight of transport variable interpolation 
        upper = ϕf * weights_f + diffusion
        lower = -ϕf * (1 - weights_f) - diffusion

        vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
    end
end

@kernel function upwind_internal(
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
    iFace = @index(Global)
    @inbounds if currentColor == colors[iFace]
        @inbounds iOwner = iOwners[iFace]
        @inbounds iNeighbor = iNeighbors[iFace]
        
        diffusion = nus[iOwner] * gDiffs[iFace]

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                      # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                                 # flux through the face
        weights_f = upwind_f(ϕf)                              # get weight of transport variable interpolation 
        upper = ϕf * weights_f + diffusion
        lower = -ϕf * (1 - weights_f) - diffusion

        @inbounds vals[ownerIdx[iFace]] += upper
        @inbounds vals[ownerRelOwnerIdx[iFace]] += lower
        @inbounds vals[neighborIdx[iFace]] += lower
        @inbounds vals[neighborRelNeighborIdx[iFace]] += upper
    end
end
