include("init.jl")
include("gpu_faceBased.jl")
include("gpu_helper.jl")
include("gpu_batchedFace.jl")
using KernelAbstractions, Atomix
"""
'abstract', as in, using `KernelAbstractions`, and not CUDA.
"""
function getFaceBasedGpuInput(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
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
    iOwners, iNeighbors, gDiffs, relativeToOwners, relativeToNbs, Sf = facesToGPUarrays(mesh.faces)
    return iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, relativeToOwners, relativeToNbs, U, Sf, bFaceValues, RHS, input.mesh.numInteriorFaces, bFaceMapping
end

function run_faceBased_abstract(
    iOwners,
    iNeighbors,
    gDiffs,
    offsets,
    nus,
    rows,
    cols,
    vals,
    relativeToOwner,
    relativeToNeighbor,
    U,
    Sf,
    bFaceValues,
    RHS,
    numInternalFaces,
    bFaceMapping,
    fused_pde
)
    backend = get_backend(iOwners)
    kernel_FusedFaceBasedAssembly(backend, 256)(
        iOwners,
        iNeighbors,
        gDiffs,
        offsets,
        nus,
        rows,
        cols,
        vals,
        relativeToOwner,
        relativeToNeighbor,
        U,
        Sf,
        bFaceValues,
        RHS,
        numInternalFaces,
        bFaceMapping,
        fused_pde;
        ndrange=length(iOwners)
    )
    KernelAbstractions.synchronize(backend)
    return rows, cols, vals, RHS
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
    @Const(relativeToOwner),
    @Const(relativeToNeighbor),
    @Const(U),
    @Const(Sf),
    @Const(bFaceValues),
    RHS,
    @Const(numInternalFaces),
    @Const(bFaceMapping),
    @Const(fused_pde)
)
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    if iNeighbor >0 
        upper, lower = 0.0, 0.0
        # upper, lower = fused_pde(U[iOwner], U[iNeighbor], Sf[iFace], nus[iOwner], gDiffs[iFace], upper, lower)

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
        relativeFaceIndex = iFace - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            diag, rhsx, rhsy, rhsz = 0.0, 0.0, 0.0, 0.0
            diag, rhsx, rhsy, rhsz = fused_pde(bFaceValues[bFaceIndex], Sf[iFace], nus[iOwner], gDiffs[iFace], diag, rhsx, rhsy, rhsz)
            # Diagonal Entry     
            idx = offsets[iOwner]
            CUDA.@atomic vals[idx] += diag

            # RHS/Source
            nCells = length(nus)
            CUDA.@atomic RHS[iOwner] += rhsx
            CUDA.@atomic RHS[iOwner+nCells] += rhsy
            CUDA.@atomic RHS[iOwner+nCells+nCells] += rhsz
        end
    end
end

function getBatchedFaceBasedGpuInput(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    numBatches, faceColors = getGreedyEdgeColoring(input)
    iOwners, iNeighbors, gDiffs, offsets, nus, rows, cols, vals, relativeToOwners, relativeToNbs, U, Sf, bFaceValues, RHS, N, bFaceMapping = getFaceBasedGpuInput(input)
    return iOwners, iNeighbors, gDiffs, offsets, nus, rows, cols, vals, relativeToOwners, relativeToNbs, U, Sf, bFaceValues, RHS, N, bFaceMapping, numBatches, input.mesh.numBoundaryFaces, faceColors
end

function run_batchedFace_abstract(
    iOwners,
    iNeighbors,
    gDiffs,
    offsets,
    nus,
    rows,
    cols,
    vals,
    relativeToOwners,
    relativeToNbs,
    U,
    Sf,
    bFaceValues,
    RHS,
    N,
    bFaceMapping,
    numBatches,
    M,
    faceColors,
    fused_pde
)
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
            relativeToNbs,
            faceColors,
            color,
            U,
            Sf,
            fused_pde;
            ndrange=N
        )
        KernelAbstractions.synchronize(backend)
    end
    kernel_abstract_boundaryFace(backend, 256)(
        iOwners,
        gDiffs,
        offsets,
        nus,
        vals,
        bFaceValues,
        RHS,
        Sf,
        bFaceMapping,
        fused_pde,
        ndrange=M
    )
    KernelAbstractions.synchronize(backend)

    return rows, cols, vals, RHS
end


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

        idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
        cols[idx] = iNeighbor
        rows[idx] = iOwner
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
    iOwners, _, gDiffs, relativeToOwners, relativeToNbs, Sf = facesToGPUarrays(input.mesh.faces)
    iFaces, iNeighbors, numInts, iFaceOffsets, numFaces = flattenCells(input)
    bFaceValues, U, bFaceMapping = gpu_getFaceValues(input)
    nCells::Int32 = length(input.mesh.cells)
    RHS = CUDA.zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(input.mesh.cells) + 2 * input.mesh.numInteriorFaces
    offsets = gpu_precalcOffsets(input)
    vals = CUDA.zeros(P, entriesNeeded)
    rows = CUDA.zeros(Int32, entriesNeeded)
    cols = CUDA.zeros(Int32, entriesNeeded)
    return iFaces, iNeighbors, numInts, nus, Sf, gDiffs, U, relativeToOwners, relativeToNbs, offsets, bFaceValues, bFaceMapping, input.mesh.numInteriorFaces, iOwners, rows, cols, vals, RHS, iFaceOffsets, numFaces
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
        start = iFaceOffsets[iCell-1] + length(input.mesh.cells[iCell-1].iFaces) 
        iFaceOffsets[iCell] = start
        iFaces[start:start+length(cell.iFaces)-1] = cell.iFaces
        iNeighbors[start:start+length(cell.iNeighbors)-1] = cell.iNeighbors
    end
    numInts = [c.nInternalFaces for c in input.mesh.cells]
    return CuArray(iFaces), CuArray(iNeighbors), CuArray(numInts), CuArray(iFaceOffsets), CuArray(nIFaces)
end

function runCellkernel(args, pde)
    backend = CUDABackend()
    kernel_FusedCellBasedAssembly(backend, 256)(args..., pde; ndrange=length(args[3]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_FusedCellBasedAssembly(
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
    fused_pde
)
    iElement = @index(Global)
    t = typeof(nus[1])
    numFaces = facesPerCell[iElement]
    nCells = length(nus)
    diag = 0.0
    startIndex = iFaceOffsets[iElement]-1
    for iFace in 1:numInteriors[iElement]
        iFaceIndex = iFaces[startIndex + iFace]
        valueUpper, valueLower = zero(t), zero(t)
        valueUpper, valueLower = fused_pde(U[iElement], U[iNeighbors[startIndex + iFace]], Sf[iFaceIndex], nus[iElement], gDiffs[iFaceIndex], valueUpper, valueLower)

        offdiag = ifelse(iOwners[iFaceIndex] == iElement, valueLower, valueUpper)

        idx = offsets[iElement] + ifelse(iOwners[iFaceIndex] == iElement, relativeToOwners[iFace], relativeToNeighbors[iFace])
        cols[idx] = iElement
        rows[idx] = iNeighbors[startIndex+iFace]
        Atomix.@atomic vals[idx] += offdiag

        diag += valueUpper
    end
    for iFace in numInteriors[iElement]+1:numFaces
        iFaceIndex = iFaces[startIndex + iFace]
        relativeFaceIndex = iFaceIndex - numInternalFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            rhsx, rhsy, rhsz = zeros(t, 3)
            diag, rhsx, rhsy, rhsz = fused_pde(bFaceValues[bFaceIndex], Sf[iFace], nus[iElement], gDiffs[iFace], diag, rhsx, rhsy, rhsz)
            RHS[iElement] += rhsx
            RHS[iElement+nCells] += rhsy
            RHS[iElement+nCells+nCells] += rhsz
        end
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end