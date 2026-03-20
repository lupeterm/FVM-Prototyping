include("init.jl")
include("gpu_faceBased.jl")
include("gpu_helper.jl")
include("gpu_batchedFace.jl")
using KernelAbstractions, Atomix
"""
'abstract', as in, using `KernelAbstractions`, and not CUDA.
"""
function getFaceBasedGpuInput(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
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
    return faces, nu, offsets, bFaceValues, U, boundaries, rows, cols, vals, RHS
end

function run_faceBased_abstract(
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
    fused_pde
)
    backend = CUDABackend()
    kernel_FusedFaceBasedAssembly(backend, 256)(
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
        fused_pde;
        ndrange=length(faces)
    )
    KernelAbstractions.synchronize(backend)
    return rows, cols, vals, RHS
end

@kernel function kernel_FusedFaceBasedAssembly(
    @Const(faces),      # CuArray{GpuFace}
    @Const(nus),
    @Const(offsets),
    @Const(bFaceValues),
    @Const(U),
    @Const(boundaries),
    rows,
    cols,
    vals,
    RHS,
    fused_pde
)
    iFace = @index(Global)
    theFace = faces[iFace]
    iOwner = theFace.iOwner
    iNeighbor = theFace.iNeighbor
    if theFace.iNeighbor != -1
        upper, lower = 0.0, 0.0
        upper, lower = fused_pde(U[iOwner], U[iNeighbor], theFace.Sf, nus[iOwner], theFace.gDiff, upper ,lower)

        idx = offsets[iOwner]
        cols[idx] = iOwner
        rows[idx] = iOwner
        Atomix.@atomic vals[idx] += upper

        idx = offsets[iOwner] + theFace.relativeToOwner
        cols[idx] = iOwner
        rows[idx] = iNeighbor
        vals[idx] += lower

        idx = offsets[iNeighbor]
        cols[idx] = iNeighbor
        rows[idx] = iNeighbor
        Atomix.@atomic vals[idx] += lower

        idx = offsets[iNeighbor] + theFace.relativeToNeighbor
        cols[idx] = iNeighbor
        rows[idx] = iOwner
        vals[idx] += upper
    else
        iBoundary = theFace.patchIndex
        if boundaries[iBoundary].isFixedValue
            relativeFaceIndex = iFace - boundaries[iBoundary].startFace
            diag, rhsx, rhsy, rhsz = 0.0, 0.0, 0.0, 0.0
            diag, rhsx, rhsy, rhsz = fused_pde(bFaceValues[relativeFaceIndex], theFace.Sf, nus[iOwner], theFace.gDiff, diag, rhsx, rhsy, rhsz)
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
end

function getBatchedFaceBasedGpuInput(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    numBatches, _= getGreedyEdgeColoring(input)
    faces, nu, offsets, bFaceValues, U, boundaries, rows, cols, vals, RHS = getFaceBasedGpuInput(input)
    return faces, nu, offsets, bFaceValues, U, boundaries, rows, cols, vals, RHS, numBatches, input.mesh.numInteriorFaces, input.mesh.numBoundaryFaces
end

function run_batchedFace_abstract(
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
    backend = CUDABackend()
    for color in 1:numBatches
        kernel_FusedBatchedFaceAssembly(backend, 256)(
            faces,      # CuArray{GpuFace}
            nus,
            offsets,
            U,
            rows,
            cols,
            vals,
            color,
            fused_pde;
            ndrange=N
        )
        KernelAbstractions.synchronize(backend)
    end
    kernel_abstract_boundaryFace(backend, 256)(
        faces,      # CuArray{GpuFace}
        nus,
        offsets,
        bFaceValues,
        boundaries,
        vals,
        RHS,
        fused_pde,
        ndrange=M
    )
    KernelAbstractions.synchronize(backend)

    return rows, cols, vals, RHS
end


@kernel function kernel_FusedBatchedFaceAssembly(
    @Const(faces),      # CuArray{GpuFace}
    @Const(nus),
    @Const(offsets),
    @Const(U),
    rows,
    cols,
    vals,
    color,
    fused_pde
)
    iFace = @index(Global)
    theFace = faces[iFace]
    iOwner = theFace.iOwner
    iNeighbor = theFace.iNeighbor

    if theFace.batchId == color[1]  # CuArray{Int32, 1, 1}
        upper, lower = 0.0, 0.0
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
    end
end


@kernel function kernel_abstract_boundaryFace(
    @Const(faces),      # CuArray{GpuFace}
    @Const(nus),
    @Const(offsets),
    @Const(bFaceValues),
    @Const(boundaries),
    vals,
    RHS,
    fused_pde
)
    t = typeof(nus[1])
    iFace = @index(Global)
    theFace = faces[iFace]
    iOwner = theFace.iOwner
    iBoundary = theFace.patchIndex
    if boundaries[iBoundary].isFixedValue
        relativeFaceIndex = iFace - boundaries[iBoundary].startFace

        diag, rhsx, rhsy, rhsz = zero(t), zero(t), zero(t), zero(t)
        diag, rhsx, rhsy, rhsz = fused_pde(bFaceValues[relativeFaceIndex], theFace.Sf, nus[iOwner], theFace.gDiff, diag, rhsx, rhsy, rhsz)

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
    return cells, faces, nu, offsets, bFaceValues, nCells, U, boundaries, rows, cols, vals, RHS
end

function runCellkernel(args...)
    backend = CUDABackend()
    kernel_FusedCellBasedAssembly(backend, 256)(args...; ndrange=length(args[1]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_FusedCellBasedAssembly(
    @Const(cells),      # CuArray{GpuCell}
    @Const(faces),      # CuArray{GpuFace}
    @Const(nus),
    @Const(offsets),
    @Const(bFaceValues),
    @Const(nCells),
    @Const(U),
    @Const(boundaries),
    rows,
    cols,
    vals,
    RHS,
    fused_pde
)
    iElement = @index(Global)
    cell = cells[iElement]
    numFaces = length(cell.iFaces)
    diag = 0.0
    cell.iNeighbors
    for iFace in 1:cell.nInternalFaces
        iFaceIndex = cell.iFaces[iFace]
        theFace = faces[iFaceIndex]

        valueUpper, valueLower = fused_pde(U[iElement], U[cell.iNeighbors[iFace]], theFace.Sf, nus[iFace], theFace.gDiff)

        offdiag = ifelse(theFace.iOwner == iElement, valueLower, valueUpper)

        idx = offsets[iElement] + ifelse(theFace.iOwner == iElement, theFace.relativeToOwner, theFace.relativeToNeighbor)
        cols[idx] = iElement
        rows[idx] = cell.iNeighbors[iFace]
        Atomix.@atomic vals[idx] += offdiag

        diag += valueUpper
    end
    for iFace in cell.nInternalFaces+1:numFaces
        iFaceIndex = cell.iFaces[iFace]
        iBoundary = faces[iFaceIndex].patchIndex
        if !boundaries[iBoundary].isFixedValue
            continue
        end
        relativeFaceIndex = iFaceIndex - boundaries[iBoundary].startFace
        theFace = faces[iFaceIndex]
        d, rhsx, rhsy, rhsz = fused_pde(bFaceValues[relativeFaceIndex], theFace.Sf, nus[iElement], theFace.gDiff)
        diag += d        
        RHS[iElement] += rhsx
        RHS[iElement+nCells] += rhsy
        RHS[iElement+nCells+nCells] += rhsz
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end