include("common.jl")


function gpu_FaceBasedAssembly(input::MatrixAssemblyInput)
    iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, numBlocks, bFaceValues, RHS, nCells, M, U, Sf, bFaceMapping = gpu_prepareFaceBased(input)
    faceBasedAllRunner(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, numBlocks, bFaceValues, RHS, nCells, M, U, Sf, bFaceMapping)
    return Vector(rows), Vector(cols), Vector(vals), Vector(RHS)
end

# div only precalc 
function gpu_DivOnlyPrecalculatedWeightsFaceBasedAssemblyRunner(
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
    kernel_DivOnlyPrecalculatedWeightsFaceBasedAssembly(backend, 64)(
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

@kernel function kernel_DivOnlyPrecalculatedWeightsFaceBasedAssembly(
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
    numFaces = length(iOwners)
    nCells = length(nus)
    numInternalFaces = numFaces - length(bFaceMapping)
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    if iNeighbor > 0
        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = weights[iFace]                          # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        upper = ϕf * weights_f
        lower = -ϕf * (1 - weights_f)


        Atomix.@atomic vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        Atomix.@atomic vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
    else
        relativeFaceIndex = iFace - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])

            # RHS/Source
            Atomix.@atomic RHS[iOwner] -= convection[1]
            Atomix.@atomic RHS[iOwner+nCells] -= convection[2]
            Atomix.@atomic RHS[iOwner+nCells+nCells] -= convection[3]
        end
    end
end


# div only hardcoded upwind
function gpu_DivOnlyHardcodedDivFaceBasedAssemblyRunner(
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
    kernel! = ifelse(upwindOrCdf == "CDF", kernel_DivOnlyHardcodedUpwindFaceBasedAssembly(backend, 64), kernel_DivOnlyHardcodedCDFFaceBasedAssembly(backend, 64))
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


@kernel function kernel_DivOnlyHardcodedUpwindFaceBasedAssembly(
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
    RHS
)
    t = eltype(nus)
    numFaces = length(iOwners)
    nCells = length(nus)
    numInternalFaces = numFaces - length(bFaceMapping)
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    if iNeighbor > 0
        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = upwind(ϕf)                          # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        upper = ϕf * weights_f
        lower = -ϕf * (1 - weights_f)


        Atomix.@atomic vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        Atomix.@atomic vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
    else
        relativeFaceIndex = iFace - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])

            # RHS/Source
            Atomix.@atomic RHS[iOwner] -= convection[1]
            Atomix.@atomic RHS[iOwner+nCells] -= convection[2]
            Atomix.@atomic RHS[iOwner+nCells+nCells] -= convection[3]
        end
    end
end



@kernel function kernel_DivOnlyHardcodedCDFFaceBasedAssembly(
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
    RHS
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
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = centralDifferencing(ϕf)                          # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        upper = ϕf * weights_f
        lower = -ϕf * (1 - weights_f)


        Atomix.@atomic vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        Atomix.@atomic vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
    else
        relativeFaceIndex = iFace - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])

            # RHS/Source
            Atomix.@atomic RHS[iOwner] -= convection[1]
            Atomix.@atomic RHS[iOwner+nCells] -= convection[2]
            Atomix.@atomic RHS[iOwner+nCells+nCells] -= convection[3]
        end
    end
end


function gpu_DivOnlyDynamicFaceBasedAssemblyRunner(
    batches,
    U,
    nus,
    bFaceValues,
    bFaceMapping,
    vals,
    RHS,
    divFunc
)
    backend = CUDABackend()
    kernel_DivOnlyDynamicFaceBasedAssembly(backend, 64)(
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
        divFunc;
        ndrange=length(batches.iOwner)
    )
    KernelAbstractions.synchronize(backend)
    return vals, RHS
end

@kernel function kernel_DivOnlyDynamicFaceBasedAssembly(
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
    t = eltype(nus)
    nCells = length(nus)
    numFaces = length(iOwners)
    numInternalFaces = numFaces - length(bFaceMapping)
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    if iNeighbor > 0
        # Convection
        Uf = 0.5(U[iOwner] + U[iNeighbor])                  # interpolate velocity to face 
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = div(ϕf)                          # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        upper = ϕf * weights_f
        lower = -ϕf * (1 - weights_f)


        Atomix.@atomic vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        Atomix.@atomic vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
    else
        relativeFaceIndex = iFace - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])

            # RHS/Source
            Atomix.@atomic RHS[iOwner] -= convection[1]
            Atomix.@atomic RHS[iOwner+nCells] -= convection[2]
            Atomix.@atomic RHS[iOwner+nCells+nCells] -= convection[3]
        end
    end
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



function gpu_LaplaceOnlyFaceBasedAssemblyRunner(
    batches,
    U,
    nus,
    bFaceValues,
    bFaceMapping,
    vals,
    RHS,
)
    backend = CUDABackend()
    kernel_LaplaceOnlyFaceBasedAssembly(backend, 64)(
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
        RHS;
        ndrange=length(batches.iOwner)
    )
    KernelAbstractions.synchronize(backend)
    return vals, RHS
end

@kernel function kernel_LaplaceOnlyFaceBasedAssembly(
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

        upper = diffusion
        lower = -diffusion

        Atomix.@atomic vals[ownerIdx[iFace]] += upper
        vals[ownerRelOwnerIdx[iFace]] += lower
        Atomix.@atomic vals[neighborIdx[iFace]] += lower
        vals[neighborRelNeighborIdx[iFace]] += upper
    else
        relativeFaceIndex = iFace - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            diffusion = nus[iOwner] * gDiffs[iFace]
            Atomix.@atomic vals[ownerIdx[iFace]] -= diffusion    # x

            # RHS/Source
            Atomix.@atomic RHS[iOwner] -= diffusion
            Atomix.@atomic RHS[iOwner+nCells] -= diffusion
            Atomix.@atomic RHS[iOwner+nCells+nCells] -= diffusion
        end
    end
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
        ϕf = dot(Uf, Sf[iFace])                    # flux through the face
        weights_f = 0.5                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        upper = ϕf * weights_f + diffusion
        lower = -ϕf * (1 - weights_f) - diffusion

        idx = offsets[iOwner]
        cols[idx] = iOwner
        rows[idx] = iOwner
        Atomix.@atomic vals[idx] += upper    # x

        idx = offsets[iOwner] + relativeToOwner[iFace]
        cols[idx] = iOwner
        rows[idx] = iNeighbor
        vals[idx] += lower    # x

        idx = offsets[iNeighbor]
        cols[idx] = iNeighbor
        rows[idx] = iNeighbor
        Atomix.@atomic vals[idx] += lower    # x

        idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
        cols[idx] = iNeighbor
        rows[idx] = iOwner
        vals[idx] += upper    # x
    else
        relativeFaceIndex = iFace - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex == -1
            return
        end
        convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])
        diffusion = nus[iOwner] * gDiffs[iFace]
        idx = offsets[iOwner]
        Atomix.@atomic vals[idx] -= diffusion    # x

        # RHS/Source
        value = convection .+ diffusion
        Atomix.@atomic RHS[iOwner] -= value[1]
        Atomix.@atomic RHS[iOwner+nCells] -= value[2]
        Atomix.@atomic RHS[iOwner+nCells+nCells] -= value[3]
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
    ϕf = dot(Uf, Sf[iFace])                    # flux through the face
    weights_f = upwind(ϕf)                              # get weight of transport variable interpolation 
    # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
    upper = ϕf * weights_f + diffusion
    lower = -ϕf * (1 - weights_f) - diffusion

    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    Atomix.@atomic vals[idx] += upper    # x

    idx = offsets[iOwner] + relativeToOwner[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] += lower    # x

    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    Atomix.@atomic vals[idx] += lower    # x

    idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += upper    # x
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
    diffusion = nus[iOwner] * gDiffs[globalFaceIndex]
    idx = offsets[iOwner]
    Atomix.@atomic vals[idx] -= diffusion    # x

    # RHS/Source
    value = convection .+ diffusion
    Atomix.@atomic RHS[iOwner] -= value[1]
    Atomix.@atomic RHS[iOwner+nCells] -= value[2]
    Atomix.@atomic RHS[iOwner+nCells+nCells] -= value[3]
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
    diffusion = nus[iOwner] * gDiffs[globalFaceIndex]
    idx = offsets[iOwner]
    Atomix.@atomic vals[idx] -= diffusion

    # RHS/Source
    Atomix.@atomic RHS[iOwner] -= diffusion
    Atomix.@atomic RHS[iOwner+nCells] -= diffusion
    Atomix.@atomic RHS[iOwner+nCells+nCells] -= diffusion
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
    Atomix.@atomic RHS[iOwner] -= convection[1]
    Atomix.@atomic RHS[iOwner+nCells] -= convection[2]
    Atomix.@atomic RHS[iOwner+nCells+nCells] -= convection[3]
    return nothing
end
