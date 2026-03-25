include("init.jl")
include("gpu_helper.jl")
using Atomix
using KernelAbstractions

function prepCellbased(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    cells = CuArray([c() for c in input.mesh.cells])
    faces = CuArray([f() for f in input.mesh.faces])
    boundaries = CuArray([b() for b in input.mesh.boundaries])
    nu = CuArray(input.nu)
    nCells::Int32 = length(input.mesh.cells)
    RHS = CUDA.zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(input.mesh.cells) + 2 * input.mesh.numInteriorFaces
    offsets = gpu_precalcOffsets(input)
    vals = CUDA.zeros(P, entriesNeeded)
    rows = CUDA.zeros(Int32, entriesNeeded)
    cols = CUDA.zeros(Int32, entriesNeeded)
    prepareRelativeIndices!(input)
    bFaceValues, U, _ = gpu_getFaceValues(input)
    return cells, faces, nu, offsets, bFaceValues, U, boundaries, rows, cols, vals, RHS
end

function gpu_LaplaceOnlyCellBasedRunner(args...)
    backend = CUDABackend()
    kernel_LaplaceOnlyCellBased(backend, 256)(args...; ndrange=length(args[3]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_LaplaceOnlyCellBased(
    @Const(iFaces),
    @Const(iNeighbors),
    @Const(numInteriors),
    @Const(nus),
    @Const(Sf),
    @Const(gDiffs),
    @Const(U),
    @Const(relativeToOwners),
    @Const(relativeToNeighbors),
    @Const(offsets),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(numInternalFaces),
    @Const(iOwners),
    rows,
    cols,
    vals,
    RHS,
    @Const(iFaceOffsets),
    @Const(facesPerCell),
)
    iElement = @index(Global)
    numFaces = facesPerCell[iElement]
    nCells = length(nus)
    diag = 0.0
    startIndex = iFaceOffsets[iElement]-1
    for iFace in 1:numInteriors[iElement]
        iFaceIndex = iFaces[startIndex+iFace]
        diffusion = nus[iElement] * gDiffs[iFaceIndex]

        valueUpper = diffusion
        valueLower = -diffusion

        offdiag = ifelse(iOwners[iFaceIndex] == iElement, valueLower, valueUpper)

        idx = offsets[iElement] + ifelse(iOwners[iFaceIndex] == iElement, relativeToOwners[iFace], relativeToNeighbors[iFace])
        cols[idx] = iElement
        rows[idx] = iNeighbors[startIndex+iFace]
        Atomix.@atomic vals[idx] += offdiag

        diag += valueUpper
    end
    for iFace in numInteriors[iElement]+1:numFaces
        iFaceIndex = iFaces[startIndex+iFace]
        relativeFaceIndex = iFaceIndex - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            diffusion = nus[iElement] * gDiffs[iFaceIndex]
            diag -= diffusion
            RHS[iElement] -= diffusion
            RHS[iElement+nCells] -= diffusion
            RHS[iElement+nCells+nCells] -= diffusion
        end
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end

function gpu_PrecalculatedWeightsCellBasedRunner(args, weights)
    backend = CUDABackend()
    kernel_PrecalculatedWeightsCellBased(backend, 256)(args..., weights; ndrange=length(args[3]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_PrecalculatedWeightsCellBased(
    @Const(iFaces),
    @Const(iNeighbors),
    @Const(numInteriors),
    @Const(nus),
    @Const(Sf),
    @Const(gDiffs),
    @Const(U),
    @Const(relativeToOwners),
    @Const(relativeToNeighbors),
    @Const(offsets),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(numInternalFaces),
    @Const(iOwners),
    rows,
    cols,
    vals,
    RHS,
    @Const(iFaceOffsets),
    @Const(facesPerCell),
    @Const(weights)
)
    iElement = @index(Global)
    numFaces = facesPerCell[iElement]
    nCells = length(nus)
    diag = 0.0
    startIndex = iFaceOffsets[iElement]-1 
    for iFace in 1:numInteriors[iElement]
        iFaceIndex = iFaces[startIndex+iFace]

        U_P = U[iElement]
        U_N = U[iNeighbors[startIndex+iFace]]
        ϕf = dot(0.5(U_P + U_N), Sf[iFaceIndex])
        weights_f = weights[iFaceIndex]
        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)

        diffusion = nus[iElement] * gDiffs[iFaceIndex]

        valueUpper += diffusion
        valueLower += -diffusion

        offdiag = ifelse(iOwners[iFaceIndex] == iElement, valueLower, valueUpper)

        idx = offsets[iElement] + ifelse(iOwners[iFaceIndex] == iElement, relativeToOwners[iFace], relativeToNeighbors[iFace])
        cols[idx] = iElement
        rows[idx] = iNeighbors[startIndex+iFace]
        Atomix.@atomic vals[idx] += offdiag

        diag += valueUpper
    end
    for iFace in numInteriors[iElement]+1:numFaces
        iFaceIndex = iFaces[startIndex+iFace]
        relativeFaceIndex = iFaceIndex - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* Sf[iFace] ⋅ bFaceValues[bFaceIndex]

            diffusion = nus[iElement] * gDiffs[iFace]
            diag -= diffusion
            RHS[iElement] -= convection - diffusion
            RHS[iElement+nCells] -= convection - diffusion
            RHS[iElement+nCells+nCells] -= convection - diffusion
        end
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end

function gpu_DynamicCellBasedRunner(args, f)
    backend = CUDABackend()
    kernel_DynamicCellBased(backend, 256)(args..., f; ndrange=length(args[3]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_DynamicCellBased(
    @Const(iFaces),
    @Const(iNeighbors),
    @Const(numInteriors),
    @Const(nus),
    @Const(Sf),
    @Const(gDiffs),
    @Const(U),
    @Const(relativeToOwners),
    @Const(relativeToNeighbors),
    @Const(offsets),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(numInternalFaces),
    @Const(iOwners),
    rows,
    cols,
    vals,
    RHS,
    @Const(iFaceOffsets),
    @Const(facesPerCell),
    @Const(divFunc)
)
    iElement = @index(Global)
    numFaces = facesPerCell[iElement] 
    nCells = length(nus)
    diag = 0.0
    startIndex = iFaceOffsets[iElement]-1
    for iFace in 1:numInteriors[iElement]
        iFaceIndex = iFaces[startIndex+iFace]

        U_P = U[iElement]
        U_N = U[iNeighbors[startIndex+iFace]]
        ϕf = dot(0.5(U_P + U_N), Sf[iFaceIndex])
        weights_f = divFunc(ϕf)
        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)

        diffusion = nus[iElement] * gDiffs[iFaceIndex]

        valueUpper += diffusion
        valueLower += -diffusion

        offdiag = ifelse(iOwners[iFaceIndex] == iElement, valueLower, valueUpper)

        idx = offsets[iElement] + ifelse(iOwners[iFaceIndex] == iElement, relativeToOwners[iFace], relativeToNeighbors[iFace])
        cols[idx] = iElement
        rows[idx] = iNeighbors[startIndex+iFace]
        Atomix.@atomic vals[idx] += offdiag

        diag += valueUpper
    end
    for iFace in numInteriors[iElement]+1:numFaces
        iFaceIndex = iFaces[startIndex+iFace]
        relativeFaceIndex = iFaceIndex - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* Sf[iFace] ⋅ bFaceValues[bFaceIndex]

            diffusion = nus[iElement] * gDiffs[iFace]
            diag -= diffusion
            RHS[iElement] -= convection - diffusion
            RHS[iElement+nCells] -= convection - diffusion
            RHS[iElement+nCells+nCells] -= convection - diffusion
        end
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end

function gpu_DivOnlyPrecalculatedWeightsCellBasedRunner(args, weights)
    backend = CUDABackend()
    kernel_DivOnlyPrecalculatedWeightsCellBased(backend, 256)(args..., weights; ndrange=length(args[3]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_DivOnlyPrecalculatedWeightsCellBased(
    @Const(iFaces),
    @Const(iNeighbors),
    @Const(numInteriors),
    @Const(nus),
    @Const(Sf),
    @Const(gDiffs),
    @Const(U),
    @Const(relativeToOwners),
    @Const(relativeToNeighbors),
    @Const(offsets),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(numInternalFaces),
    @Const(iOwners),
    rows,
    cols,
    vals,
    RHS,
    @Const(iFaceOffsets),
    @Const(facesPerCell),
    @Const(weights)
)
    iElement = @index(Global)
    numFaces = facesPerCell[iElement]
    nCells = length(nus)
    diag = 0.0
    startIndex = iFaceOffsets[iElement]-1
    for iFace in 1:numInteriors[iElement]
        iFaceIndex = iFaces[startIndex+iFace]

        U_P = U[iElement]
        U_N = U[iNeighbors[startIndex+iFace]]
        ϕf = dot(0.5(U_P + U_N), Sf[iFaceIndex])
        weights_f = weights[iFaceIndex]
        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)



        offdiag = ifelse(iOwners[iFaceIndex] == iElement, valueLower, valueUpper)

        idx = offsets[iElement] + ifelse(iOwners[iFaceIndex] == iElement, relativeToOwners[iFace], relativeToNeighbors[iFace])
        cols[idx] = iElement
        rows[idx] = iNeighbors[startIndex+iFace]
        Atomix.@atomic vals[idx] += offdiag

        diag += valueUpper
    end
    for iFace in numInteriors[iElement]+1:numFaces
        iFaceIndex = iFaces[startIndex+iFace]
        relativeFaceIndex = iFaceIndex - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* Sf[iFace] ⋅ bFaceValues[bFaceIndex]

            RHS[iElement] -= convection
            RHS[iElement+nCells] -= convection
            RHS[iElement+nCells+nCells] -= convection
        end
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end

function gpu_DivOnlyHardcodedUpwindCellBasedRunner(args...)
    backend = CUDABackend()
    kernel_DivOnlyHardCodedUpwindCellBased(backend, 256)(args...; ndrange=length(args[3]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_DivOnlyHardCodedUpwindCellBased(
    @Const(iFaces),
    @Const(iNeighbors),
    @Const(numInteriors),
    @Const(nus),
    @Const(Sf),
    @Const(gDiffs),
    @Const(U),
    @Const(relativeToOwners),
    @Const(relativeToNeighbors),
    @Const(offsets),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(numInternalFaces),
    @Const(iOwners),
    rows,
    cols,
    vals,
    RHS,
    @Const(iFaceOffsets),
    @Const(facesPerCell),
)
    iElement = @index(Global)
    numFaces = facesPerCell[iElement]
    nCells = length(nus)
    diag = 0.0
    startIndex = iFaceOffsets[iElement]-1
    for iFace in 1:numInteriors[iElement]
        iFaceIndex = iFaces[startIndex+iFace]

        U_P = U[iElement]
        U_N = U[iNeighbors[startIndex+iFace]]
        ϕf = dot(0.5(U_P + U_N), Sf[iFaceIndex])
        weights_f = upwind(ϕf)
        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)

        offdiag = ifelse(iOwners[iFaceIndex] == iElement, valueLower, valueUpper)

        idx = offsets[iElement] + ifelse(iOwners[iFaceIndex] == iElement, relativeToOwners[iFace], relativeToNeighbors[iFace])
        cols[idx] = iElement
        rows[idx] = iNeighbors[startIndex+iFace]
        Atomix.@atomic vals[idx] += offdiag

        diag += valueUpper
    end
    for iFace in numInteriors[iElement]+1:numFaces
        iFaceIndex = iFaces[startIndex+iFace]
        relativeFaceIndex = iFaceIndex - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* Sf[iFace] ⋅ bFaceValues[bFaceIndex]

            RHS[iElement] -= convection
            RHS[iElement+nCells] -= convection
            RHS[iElement+nCells+nCells] -= convection
        end
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end

function gpu_DivOnlyHardcodedCDFCellBasedRunner(args...)
    backend = CUDABackend()
    kernel_DivOnlyHardCodedCDFCellBased(backend, 256)(args...; ndrange=length(args[3]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_DivOnlyHardCodedCDFCellBased(
    @Const(iFaces),
    @Const(iNeighbors),
    @Const(numInteriors),
    @Const(nus),
    @Const(Sf),
    @Const(gDiffs),
    @Const(U),
    @Const(relativeToOwners),
    @Const(relativeToNeighbors),
    @Const(offsets),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(numInternalFaces),
    @Const(iOwners),
    rows,
    cols,
    vals,
    RHS,
    @Const(iFaceOffsets),
    @Const(facesPerCell),
)
    iElement = @index(Global)
    numFaces = facesPerCell[iElement]
    nCells = length(nus)
    diag = 0.0
    startIndex = iFaceOffsets[iElement]-1
    for iFace in 1:numInteriors[iElement]
        iFaceIndex = iFaces[startIndex+iFace]

        U_P = U[iElement]
        U_N = U[iNeighbors[startIndex+iFace]]
        ϕf = dot(0.5(U_P + U_N), Sf[iFaceIndex])
        weights_f = centralDifferencing(ϕf)
        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)

        offdiag = ifelse(iOwners[iFaceIndex] == iElement, valueLower, valueUpper)

        idx = offsets[iElement] + ifelse(iOwners[iFaceIndex] == iElement, relativeToOwners[iFace], relativeToNeighbors[iFace])
        cols[idx] = iElement
        rows[idx] = iNeighbors[startIndex+iFace]
        Atomix.@atomic vals[idx] += offdiag

        diag += valueUpper
    end
    for iFace in numInteriors[iElement]+1:numFaces
        iFaceIndex = iFaces[startIndex+iFace]
        relativeFaceIndex = iFaceIndex - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* Sf[iFace] ⋅ bFaceValues[bFaceIndex]

            RHS[iElement] -= convection
            RHS[iElement+nCells] -= convection
            RHS[iElement+nCells+nCells] -= convection
        end
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end

function gpu_DivOnlyDynamicCellBasedRunner(args, f)
    backend = CUDABackend()
    kernel_DivOnlyDynamicCellBased(backend, 256)(args..., f; ndrange=length(args[3]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_DivOnlyDynamicCellBased(
    @Const(iFaces),
    @Const(iNeighbors),
    @Const(numInteriors),
    @Const(nus),
    @Const(Sf),
    @Const(gDiffs),
    @Const(U),
    @Const(relativeToOwners),
    @Const(relativeToNeighbors),
    @Const(offsets),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(numInternalFaces),
    @Const(iOwners),
    rows,
    cols,
    vals,
    RHS,
    @Const(iFaceOffsets),
    @Const(facesPerCell),
    @Const(divFunc)
)
    iElement = @index(Global)
    numFaces = facesPerCell[iElement]
    nCells = length(nus)
    diag = 0.0
    startIndex = iFaceOffsets[iElement]-1
    for iFace in 1:numFaces
        iFaceIndex = iFaces[startIndex+iFace]
        if iNeighbors[startIndex+iFace] > 0

            U_P = U[iElement]
            U_N = U[iNeighbors[startIndex+iFace]]
            ϕf = dot(0.5(U_P + U_N), Sf[iFaceIndex])
            weights_f = divFunc(ϕf)
            valueUpper = ϕf * weights_f
            valueLower = -ϕf * (1 - weights_f)

            offdiag = ifelse(iOwners[iFaceIndex] == iElement, valueLower, valueUpper)

            idx = offsets[iElement] + ifelse(iOwners[iFaceIndex] == iElement, relativeToOwners[iFace], relativeToNeighbors[iFace])
            cols[idx] = iElement
            rows[idx] = iNeighbors[startIndex+iFace]
            Atomix.@atomic vals[idx] += offdiag

            diag += valueUpper
        else
            relativeFaceIndex = iFaceIndex - numInternalFaces
            bFaceIndex = bFaceMapping[relativeFaceIndex]
            if bFaceIndex != -1 && bFaceIndex < length(bFaceValues)
                convection = bFaceValues[bFaceIndex] .* Sf[iFace] ⋅ bFaceValues[bFaceIndex]
                
                RHS[iElement] -= convection
                RHS[iElement+nCells] -= convection
                RHS[iElement+nCells+nCells] -= convection
            end
        end
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end

function gpu_HardcodedUpwindCellBasedRunner(args...)
    backend = CUDABackend()
    kernel_HardCodedUpwindCellBased(backend, 256)(args...; ndrange=length(args[3]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_HardCodedUpwindCellBased(
    @Const(iFaces),
    @Const(iNeighbors),
    @Const(numInteriors),
    @Const(nus),
    @Const(Sf),
    @Const(gDiffs),
    @Const(U),
    @Const(relativeToOwners),
    @Const(relativeToNeighbors),
    @Const(offsets),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(numInternalFaces),
    @Const(iOwners),
    rows,
    cols,
    vals,
    RHS,
    @Const(iFaceOffsets),
    @Const(facesPerCell),
)
    iElement = @index(Global)
    numFaces = facesPerCell[iElement]
    nCells = length(nus)
    diag = 0.0
    startIndex = iFaceOffsets[iElement]-1
    for iFace in 1:numInteriors[iElement]
        iFaceIndex = iFaces[startIndex+iFace]

        U_P = U[iElement]
        U_N = U[iNeighbors[startIndex+iFace]]
        ϕf = dot(0.5(U_P + U_N), Sf[iFaceIndex])
        weights_f = upwind(ϕf)
        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)

        diffusion = nus[iElement] * gDiffs[iFaceIndex]

        valueUpper += diffusion
        valueLower += -diffusion

        offdiag = ifelse(iOwners[iFaceIndex] == iElement, valueLower, valueUpper)

        idx = offsets[iElement] + ifelse(iOwners[iFaceIndex] == iElement, relativeToOwners[iFace], relativeToNeighbors[iFace])
        cols[idx] = iElement
        rows[idx] = iNeighbors[startIndex+iFace]
        Atomix.@atomic vals[idx] += offdiag

        diag += valueUpper
    end
    for iFace in numInteriors[iElement]+1:numFaces
        iFaceIndex = iFaces[startIndex+iFace]
        relativeFaceIndex = iFaceIndex - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* Sf[iFace] ⋅ bFaceValues[bFaceIndex]

            diffusion = nus[iElement] * gDiffs[iFace]
            diag -= diffusion
            RHS[iElement] -= convection - diffusion
            RHS[iElement+nCells] -= convection - diffusion
            RHS[iElement+nCells+nCells] -= convection - diffusion
        end
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end

function gpu_HardcodedCDFCellBasedRunner(args...)
    backend = CUDABackend()
    kernel_HardCodedCDFCellBased(backend, 256)(args...; ndrange=length(args[3]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_HardCodedCDFCellBased(
    @Const(iFaces),
    @Const(iNeighbors),
    @Const(numInteriors),
    @Const(nus),
    @Const(Sf),
    @Const(gDiffs),
    @Const(U),
    @Const(relativeToOwners),
    @Const(relativeToNeighbors),
    @Const(offsets),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(numInternalFaces),
    @Const(iOwners),
    rows,
    cols,
    vals,
    RHS,
    @Const(iFaceOffsets),
    @Const(facesPerCell),
)
    iElement = @index(Global)
    numFaces = facesPerCell[iElement]
    nCells = length(nus)
    diag = 0.0
    startIndex = iFaceOffsets[iElement]-1
    for iFace in 1:numInteriors[iElement]
        iFaceIndex = iFaces[startIndex+iFace]

        U_P = U[iElement]
        U_N = U[iNeighbors[startIndex+iFace]]
        ϕf = dot(0.5(U_P + U_N), Sf[iFaceIndex])
        weights_f = centralDifferencing(ϕf)
        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)

        diffusion = nus[iElement] * gDiffs[iFaceIndex]

        valueUpper += diffusion
        valueLower += -diffusion

        offdiag = ifelse(iOwners[iFaceIndex] == iElement, valueLower, valueUpper)

        idx = offsets[iElement] + ifelse(iOwners[iFaceIndex] == iElement, relativeToOwners[iFace], relativeToNeighbors[iFace])
        cols[idx] = iElement
        rows[idx] = iNeighbors[startIndex+iFace]
        Atomix.@atomic vals[idx] += offdiag

        diag += valueUpper
    end
    for iFace in numInteriors[iElement]+1:numFaces
        iFaceIndex = iFaces[startIndex+iFace]
        relativeFaceIndex = iFaceIndex - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* Sf[iFace] ⋅ bFaceValues[bFaceIndex]

            diffusion = nus[iElement] * gDiffs[iFace]
            diag -= diffusion
            RHS[iElement] -= convection - diffusion
            RHS[iElement+nCells] -= convection - diffusion
            RHS[iElement+nCells+nCells] -= convection - diffusion
        end
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end