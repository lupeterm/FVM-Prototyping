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
    kernel_PrecalculatedWeightsFaceBasedAssembly(backend, 64)(
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
        vals,
        RHS,
        weights;
        ndrange=length(batches.iOwner)
    )
    KernelAbstractions.synchronize(backend)
    return vals, RHS
end

@kernel function kernel_PrecalculatedWeightsFaceBasedAssembly(
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
    vals,
    RHS,
    @Const(weights)
)
    t = eltype(nus)
    nCells = length(nus)
    numFaces = length(iOwners)
    numInternalFaces = numFaces - length(bFaceMapping)
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    if iNeighbor > 0
        # Convection
        # Diffusion
        diffusion = nus[iOwner] * gDiffs[iFace]

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = weights[iFace]                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        upper = ϕf * weights_f + diffusion
        lower = -ϕf * (1 - weights_f) - diffusion


        Atomix.@atomic vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        Atomix.@atomic vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
    else
        relativeFaceIndex = iFace - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])
            diffusion = nus[iOwner] * gDiffs[iFace]
            Atomix.@atomic vals[ownerIdx[iFace]] -= diffusion    # x

            # RHS/Source
            Atomix.@atomic RHS[iOwner] -= convection[1] - diffusion
            Atomix.@atomic RHS[iOwner+nCells] -= convection[2] - diffusion
            Atomix.@atomic RHS[iOwner+nCells+nCells] -= convection[3] - diffusion
        end
    end
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
    kernel_fusedFaceBasedAssembly(backend, 64)(
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
        vals,
        RHS,
        pde;
        ndrange=length(batches.iOwner)
    )
    KernelAbstractions.synchronize(backend)
    return vals, RHS
end

@kernel function kernel_fusedFaceBasedAssembly(
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
    vals,
    RHS,
    @Const(fused_pde)
)
    t = eltype(nus)
    nCells = length(nus)
    numFaces = length(iOwners)
    numInternalFaces = numFaces - length(bFaceMapping)
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    if iNeighbor > 0
        valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbors[iFace]], Sf[iFace], nus[iOwner], gDiffs[iFace], 0.0, 0.0)

        Atomix.@atomic vals[ownerIdx[iFace]] += valueUpper
        vals[ownerRelOwnerIdx[iFace]] += valueLower
        Atomix.@atomic vals[neighborIdx[iFace]] += valueLower
        vals[neighborRelNeighborIdx[iFace]] += valueUpper
    else
        relativeFaceIndex = iFace - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])
            diffusion = nus[iOwner] * gDiffs[iFace]
            Atomix.@atomic vals[ownerIdx[iFace]] -= diffusion    # x

            # RHS/Source
            Atomix.@atomic RHS[iOwner] -= convection[1] - diffusion
            Atomix.@atomic RHS[iOwner+nCells] -= convection[2] - diffusion
            Atomix.@atomic RHS[iOwner+nCells+nCells] -= convection[3] - diffusion
        end
    end
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
    kernel! = ifelse(upwindOrCdf == "CDF", kernel_HardcodedUpwindFaceBasedAssembly(backend, 64), kernel_HardcodedCDFFaceBasedAssembly(backend, 64))
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
        bFaceValues,
        bFaceMapping,
        vals,
        RHS,
        ndrange=length(batches.iOwner)
    )
    KernelAbstractions.synchronize(backend)
    return vals, RHS
end

@kernel function kernel_HardcodedUpwindFaceBasedAssembly(
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
    vals,
    RHS,
)
    numFaces = length(iOwners)
    nCells = length(nus)
    numInternalFaces = numFaces - length(bFaceMapping)
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    if iNeighbor > 0
        # Convection
        # Diffusion
        diffusion = nus[iOwner] * gDiffs[iFace]

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = upwind(ϕf)                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        upper = ϕf * weights_f + diffusion
        lower = -ϕf * (1 - weights_f) - diffusion


        Atomix.@atomic vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        Atomix.@atomic vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
    else
        relativeFaceIndex = iFace - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])
            diffusion = nus[iOwner] * gDiffs[iFace]
            Atomix.@atomic vals[ownerIdx[iFace]] -= diffusion    # x

            # RHS/Source
            Atomix.@atomic RHS[iOwner] -= convection[1] - diffusion
            Atomix.@atomic RHS[iOwner+nCells] -= convection[2] - diffusion
            Atomix.@atomic RHS[iOwner+nCells+nCells] -= convection[3] - diffusion
        end
    end
end


@kernel function kernel_HardcodedCDFFaceBasedAssembly(
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
    vals,
    RHS,
)
    numFaces = length(iOwners)
    nCells = length(nus)
    numInternalFaces = numFaces - length(bFaceMapping)
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    if iNeighbor > 0
        # Convection
        # Diffusion
        diffusion = nus[iOwner] * gDiffs[iFace]

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = centralDifferencing(ϕf)                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        upper = ϕf * weights_f + diffusion
        lower = -ϕf * (1 - weights_f) - diffusion


        Atomix.@atomic vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        Atomix.@atomic vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
    else
        relativeFaceIndex = iFace - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])
            diffusion = nus[iOwner] * gDiffs[iFace]
            Atomix.@atomic vals[ownerIdx[iFace]] -= diffusion    # x

            # RHS/Source
            Atomix.@atomic RHS[iOwner] -= convection[1] - diffusion
            Atomix.@atomic RHS[iOwner+nCells] -= convection[2] - diffusion
            Atomix.@atomic RHS[iOwner+nCells+nCells] -= convection[3] - diffusion
        end
    end
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
    kernel_DynamicFaceBasedAssembly(backend, 64)(
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
        bFaceValues,
        bFaceMapping,
        vals,
        RHS,
        divFunc;
        ndrange=length(faces.iOwner)
    )
    KernelAbstractions.synchronize(backend)
    return vals, RHS
end

@kernel function kernel_DynamicFaceBasedAssembly(
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
    vals,
    RHS,
    @Const(div)
)
    numFaces = length(iOwners)
    nCells = length(nus)
    numInternalFaces = numFaces - length(bFaceMapping)
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    if iNeighbor > 0
        # Convection
        # Diffusion
        diffusion = nus[iOwner] * gDiffs[iFace]

        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = div(ϕf)                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        upper = ϕf * weights_f + diffusion
        lower = -ϕf * (1 - weights_f) - diffusion


        Atomix.@atomic vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        Atomix.@atomic vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
    else
        relativeFaceIndex = iFace - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])
            diffusion = nus[iOwner] * gDiffs[iFace]
            Atomix.@atomic vals[ownerIdx[iFace]] -= diffusion    # x

            # RHS/Source
            Atomix.@atomic RHS[iOwner] -= convection[1] - diffusion
            Atomix.@atomic RHS[iOwner+nCells] -= convection[2] - diffusion
            Atomix.@atomic RHS[iOwner+nCells+nCells] -= convection[3] - diffusion
        end
    end
end



