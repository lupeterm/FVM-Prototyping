include("init.jl")
include("gpu_helper.jl")
using Atomix
using KernelAbstractions

@kernel function test(bs)
    i = @index(Global)
    cell = bs[i]
    if cell.isFixedValue
        @print("yes")
    else
        @print("noe")
    end

end

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
    return cells, faces, nu, offsets, bFaceValues, nCells, U, boundaries, rows, cols, vals, RHS
end

function runCellkernel(args...)
    backend = CUDABackend()
    kernel_AllCellBasedAssemblyRunner(backend, 256)(args...; ndrange=length(args[1]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_AllCellBasedAssemblyRunner(
    @Const(cells), # CuArray{GpuCell}
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
    ops
)
    iElement = @index(Global)
    cell = cells[iElement]
    numFaces = length(cell.iFaces)
    diag = 0.0
    cell.iNeighbors
    for iFace in 1:cell.nInternalFaces
        iFaceIndex = cell.iFaces[iFace]
        theFace = faces[iFaceIndex]

        valueUpper = 0.0
        valueLower = 0.0

        valueUpper, valueLower = ops(U[iElement], U[cell.iNeighbors[iFace]], theFace.Sf, nus[iFace], theFace.gDiff, valueUpper, valueLower)

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
        theFace = faces[iFaceIndex]
        relativeFaceIndex = iFaceIndex - boundaries[iBoundary].startFace
        ϕf = theFace.Sf ⋅ bFaceValues[relativeFaceIndex]
        convection = bFaceValues[relativeFaceIndex] .* ϕf
        diffusion = nus[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        diag -= diffusion

        RHS[iElement] -= convection[1] - diffusion
        RHS[iElement+nCells] -= convection[2] - diffusion
        RHS[iElement+nCells+nCells] -= convection[3] - diffusion
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end