include("common.jl")


function gpu_FaceBasedAssembly(input::MatrixAssemblyInput)
    iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, numBlocks, bFaceValues, RHS, nCells, M, U, Sf, bFaceMapping = gpu_prepareFaceBased(input)
    faceBasedAllRunner(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, numBlocks, bFaceValues, RHS, nCells, M, U, Sf, bFaceMapping)
    return Vector(rows), Vector(cols), Vector(vals), Vector(RHS)
end


function gpu_PrecalculatedWeightsFaceBasedAssemblyRunner(
    batches,
    U,
    nus,
    bFaceValues,
    bFaceMapping,
    vals,
    RHS,
    weights
)
    backend = CUDABackend()
    numInternalFaces = length(batches.iOwner) - length(bFaceMapping)
    kernel_PrecalculatedWeightsFaceBasedAssembly_internal(backend, 64)(
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
        vals,
        weights;
        ndrange=numInternalFaces
    )
    KernelAbstractions.synchronize(backend)
    kernel_boundary_nonfused(backend, 64)(
        batches.iOwner,
        batches.gDiff,
        batches.ownerIdx,
        batches.Sf,
        nus,
        bFaceValues,
        bFaceMapping,
        vals,
        RHS,
        numInternalFaces;
        ndrange=length(bFaceMapping)
    )
    KernelAbstractions.synchronize(backend)
    return vals, RHS
end

@kernel function kernel_PrecalculatedWeightsFaceBasedAssembly_internal(
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
    vals,
    @Const(weights)
)
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]

    diffusion = nus[iOwner] * gDiffs[iFace]

    Uf = 0.5(U[iOwner] + U[iNeighbor])
    ϕf = dot(Uf, Sf[iFace])
    weights_f = weights[iFace]
    upper = ϕf * weights_f + diffusion
    lower = -ϕf * (1 - weights_f) - diffusion

    Atomix.@atomic vals[ownerIdx[iFace]] += upper
    vals[ownerRelOwnerIdx[iFace]] += lower
    Atomix.@atomic vals[neighborIdx[iFace]] += lower
    vals[neighborRelNeighborIdx[iFace]] += upper
end


function gpu_fusedFaceBasedAssemblyRunner(
    batches,
    U,
    nus,
    bFaceValues,
    bFaceMapping,
    vals,
    RHS,
    pde
)
    backend = CUDABackend()
    numInternalFaces = length(batches.iOwner) - length(bFaceMapping)
    kernel_fusedFaceBasedAssembly_internal(backend, 64)(
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
        vals,
        pde;
        ndrange=numInternalFaces
    )
    KernelAbstractions.synchronize(backend)
    kernel_boundary_fused(backend, 64)(
        batches.iOwner,
        batches.gDiff,
        batches.ownerIdx,
        batches.Sf,
        nus,
        bFaceValues,
        bFaceMapping,
        vals,
        RHS,
        pde,
        numInternalFaces;
        ndrange=length(bFaceMapping)
    )
    KernelAbstractions.synchronize(backend)
    return vals, RHS
end

@kernel function kernel_fusedFaceBasedAssembly_internal(
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
    vals,
    @Const(fused_pde)
)
    t = eltype(nus)
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]

    valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], Sf[iFace], nus[iOwner], gDiffs[iFace], zero(t), zero(t))

    Atomix.@atomic vals[ownerIdx[iFace]] += valueUpper
    vals[ownerRelOwnerIdx[iFace]] += valueLower
    Atomix.@atomic vals[neighborIdx[iFace]] += valueLower
    vals[neighborRelNeighborIdx[iFace]] += valueUpper
end

function gpu_HardcodedFaceBasedAssemblyRunner(
    batches,
    U,
    nus,
    bFaceValues,
    bFaceMapping,
    vals,
    RHS,
    upwindOrCdf::String
)
    backend = CUDABackend()
    numInternalFaces = length(batches.iOwner) - length(bFaceMapping)
    kernel! = ifelse(upwindOrCdf == "CDF", kernel_HardcodedCDFFaceBasedAssembly_internal(backend, 64), kernel_HardcodedUpwindFaceBasedAssembly_internal(backend, 64))
    kernel!(
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
        vals;
        ndrange=numInternalFaces
    )
    KernelAbstractions.synchronize(backend)
    kernel_boundary_nonfused(backend, 64)(
        batches.iOwner,
        batches.gDiff,
        batches.ownerIdx,
        batches.Sf,
        nus,
        bFaceValues,
        bFaceMapping,
        vals,
        RHS,
        numInternalFaces;
        ndrange=length(bFaceMapping)
    )
    KernelAbstractions.synchronize(backend)
    return vals, RHS
end

@kernel function kernel_HardcodedUpwindFaceBasedAssembly_internal(
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
    vals,
)
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]

    diffusion = nus[iOwner] * gDiffs[iFace]

    Uf = 0.5(U[iOwner] + U[iNeighbor])
    ϕf = dot(Uf, Sf[iFace])
    weights_f = upwind(ϕf)
    upper = ϕf * weights_f + diffusion
    lower = -ϕf * (1 - weights_f) - diffusion

    Atomix.@atomic vals[ownerIdx[iFace]] += upper
    vals[ownerRelOwnerIdx[iFace]] += lower
    Atomix.@atomic vals[neighborIdx[iFace]] += lower
    vals[neighborRelNeighborIdx[iFace]] += upper
end


@kernel function kernel_HardcodedCDFFaceBasedAssembly_internal(
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
    vals,
)
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]

    diffusion = nus[iOwner] * gDiffs[iFace]

    Uf = 0.5(U[iOwner] + U[iNeighbor])
    ϕf = dot(Uf, Sf[iFace])
    weights_f = centralDifferencing(ϕf)
    upper = ϕf * weights_f + diffusion
    lower = -ϕf * (1 - weights_f) - diffusion

    Atomix.@atomic vals[ownerIdx[iFace]] += upper
    vals[ownerRelOwnerIdx[iFace]] += lower
    Atomix.@atomic vals[neighborIdx[iFace]] += lower
    vals[neighborRelNeighborIdx[iFace]] += upper
end

function gpu_DynamicFaceBasedAssemblyRunner(
    faces,
    U,
    nus,
    bFaceValues,
    bFaceMapping,
    vals,
    RHS,
    divFunc::Function
)
    backend = CUDABackend()
    numInternalFaces = length(faces.iOwner) - length(bFaceMapping)
    kernel_DynamicFaceBasedAssembly_internal(backend, 64)(
        faces.iOwner,
        faces.iNeighbor,
        faces.gDiff,
        faces.ownerIdx,
        faces.ownerRelOwnerIdx,
        faces.neighborIdx,
        faces.neighborRelNeighborIdx,
        faces.Sf,
        nus,
        U,
        vals,
        divFunc;
        ndrange=numInternalFaces
    )
    KernelAbstractions.synchronize(backend)
    kernel_boundary_nonfused(backend, 64)(
        faces.iOwner,
        faces.gDiff,
        faces.ownerIdx,
        faces.Sf,
        nus,
        bFaceValues,
        bFaceMapping,
        vals,
        RHS,
        numInternalFaces;
        ndrange=length(bFaceMapping)
    )
    KernelAbstractions.synchronize(backend)
    return vals, RHS
end

@kernel function kernel_DynamicFaceBasedAssembly_internal(
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
    vals,
    @Const(div)
)
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]

    diffusion = nus[iOwner] * gDiffs[iFace]

    Uf = 0.5(U[iOwner] + U[iNeighbor])
    ϕf = dot(Uf, Sf[iFace])
    weights_f = div(ϕf)
    upper = ϕf * weights_f + diffusion
    lower = -ϕf * (1 - weights_f) - diffusion

    Atomix.@atomic vals[ownerIdx[iFace]] += upper
    vals[ownerRelOwnerIdx[iFace]] += lower
    Atomix.@atomic vals[neighborIdx[iFace]] += lower
    vals[neighborRelNeighborIdx[iFace]] += upper
end

@kernel function kernel_boundary_nonfused(
    @Const(iOwners),
    @Const(gDiffs),
    @Const(ownerIdx),
    @Const(Sf),
    @Const(nus),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    vals,
    RHS,
    @Const(numInternalFaces)
)
    nCells = length(nus)
    relativeFaceIndex = @index(Global)
    iFace = numInternalFaces + relativeFaceIndex
    iOwner = iOwners[iFace]
    bFaceIndex = bFaceMapping[relativeFaceIndex]
    if bFaceIndex != -1
        convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])
        diffusion = nus[iOwner] * gDiffs[iFace]
        Atomix.@atomic vals[ownerIdx[iFace]] -= diffusion

        Atomix.@atomic RHS[iOwner] -= convection[1] - diffusion
        Atomix.@atomic RHS[iOwner+nCells] -= convection[2] - diffusion
        Atomix.@atomic RHS[iOwner+nCells+nCells] -= convection[3] - diffusion
    end
end

@kernel function kernel_boundary_fused(
    @Const(iOwners),
    @Const(gDiffs),
    @Const(ownerIdx),
    @Const(Sf),
    @Const(nus),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    vals,
    RHS,
    @Const(fused_pde),
    @Const(numInternalFaces)
)
    t = eltype(nus)
    nCells = length(nus)
    relativeFaceIndex = @index(Global)
    iFace = numInternalFaces + relativeFaceIndex
    iOwner = iOwners[iFace]
    bFaceIndex = bFaceMapping[relativeFaceIndex]
    if bFaceIndex != -1
        diag, rhsx, rhsy, rhsz = fused_pde(bFaceValues[bFaceIndex], Sf[iFace], nus[iOwner], gDiffs[iFace], zero(t), zero(t), zero(t), zero(t))

        Atomix.@atomic vals[ownerIdx[iFace]] += diag
        Atomix.@atomic RHS[iOwner] += rhsx
        Atomix.@atomic RHS[iOwner+nCells] += rhsy
        Atomix.@atomic RHS[iOwner+nCells+nCells] += rhsz
    end
end
