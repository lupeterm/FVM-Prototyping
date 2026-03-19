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

function LaplaceOnlyCellBasedRunner(args...)
    backend = CUDABackend()
    kernel_LaplaceOnlyCellBased(backend, 256)(args...; ndrange=length(args[1]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_LaplaceOnlyCellBased(
    @Const(cells),      # CuArray{GpuCell}
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
)
    iElement = @index(Global)
    cell = cells[iElement]
    numFaces = length(cell.iFaces)
    diag = 0.0
    nCells = length(cells)
    for iFace in 1:cell.nInternalFaces
        iFaceIndex = cell.iFaces[iFace]
        theFace = faces[iFaceIndex]

        diffusion = nus[iElement] * theFace.gDiff

        valueUpper = diffusion
        valueLower = -diffusion

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
        diffusion = nus[iElement] * theFace.gDiff

        diag -= diffusion
        RHS[iElement] -= diffusion
        RHS[iElement+nCells] -= diffusion
        RHS[iElement+nCells+nCells] -= diffusion
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end

function PrecalculatedWeightsCellBasedRunner(args, weights)
    backend = CUDABackend()
    kernel_PrecalculatedWeightsCellBased(backend, 256)(args..., weights; ndrange=length(args[1]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_PrecalculatedWeightsCellBased(
    @Const(cells),      # CuArray{GpuCell}
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
    @Const(weights)
)
    iElement = @index(Global)
    cell = cells[iElement]
    numFaces = length(cell.iFaces)
    diag = 0.0
    nCells = length(cells)

    for iFace in 1:cell.nInternalFaces
        iFaceIndex = cell.iFaces[iFace]
        theFace = faces[iFaceIndex]

        diffusion = nus[iElement] * theFace.gDiff

        # Convection
        Uf = 0.5(U[iElement] + U[cell.iNeighbors[iFace]])                  # interpolate velocity to face 
        ϕf = dot(Uf, theFace.Sf)                    # flux through the face
        weights_f = weights[iFace]                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = ϕf * weights_f + diffusion
        valueLower = -ϕf * (1 - weights_f) - diffusion

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
        convection = bFaceValues[relativeFaceIndex] .* dot(theFace.Sf, bFaceValues[relativeFaceIndex])
        diffusion = nus[iElement] * theFace.gDiff

        diag -= diffusion
        RHS[iElement] -= diffusion - convection[1]
        RHS[iElement+nCells] -= diffusion - convection[2]
        RHS[iElement+nCells+nCells] -= diffusion - convection[3]
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end

function DynamicCellBasedRunner(args, f)
    backend = CUDABackend()
    kernel_DynamicCellBased(backend, 256)(args..., f; ndrange=length(args[1]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_DynamicCellBased(
    @Const(cells),      # CuArray{GpuCell}
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
    @Const(divFunc)
)
    iElement = @index(Global)
    cell = cells[iElement]
    numFaces = length(cell.iFaces)
    diag = 0.0
    nCells = length(cells)

    for iFace in 1:cell.nInternalFaces
        iFaceIndex = cell.iFaces[iFace]
        theFace = faces[iFaceIndex]

        diffusion = nus[iElement] * theFace.gDiff

        # Convection
        Uf = 0.5(U[iElement] + U[cell.iNeighbors[iFace]])                  # interpolate velocity to face 
        ϕf = dot(Uf, theFace.Sf)                    # flux through the face
        weights_f = divFunc(ϕf)                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = ϕf * weights_f + diffusion
        valueLower = -ϕf * (1 - weights_f) - diffusion

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
        convection = bFaceValues[relativeFaceIndex] .* dot(theFace.Sf, bFaceValues[relativeFaceIndex])
        diffusion = nus[iElement] * theFace.gDiff

        diag -= diffusion
        RHS[iElement] -= diffusion - convection[1]
        RHS[iElement+nCells] -= diffusion - convection[2]
        RHS[iElement+nCells+nCells] -= diffusion - convection[3]
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end

function DivOnlyPrecalculatedWeightsCellBasedRunner(args, weights)
    backend = CUDABackend()
    kernel_DivOnlyPrecalculatedWeightsCellBased(backend, 256)(args..., weights; ndrange=length(args[1]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_DivOnlyPrecalculatedWeightsCellBased(
    @Const(cells),      # CuArray{GpuCell}
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
    @Const(weights)
)
    iElement = @index(Global)
    cell = cells[iElement]
    numFaces = length(cell.iFaces)
    diag = 0.0
    nCells = length(cells)

    for iFace in 1:cell.nInternalFaces
        iFaceIndex = cell.iFaces[iFace]
        theFace = faces[iFaceIndex]

        # Convection
        Uf = 0.5(U[iElement] + U[cell.iNeighbors[iFace]])                  # interpolate velocity to face 
        ϕf = dot(Uf, theFace.Sf)                    # flux through the face
        weights_f = weights[iFace]                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)

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
        convection = bFaceValues[relativeFaceIndex] .* dot(theFace.Sf, bFaceValues[relativeFaceIndex])

        RHS[iElement] -= convection[1]
        RHS[iElement+nCells] -= convection[2]
        RHS[iElement+nCells+nCells] -= convection[3]
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end

function DivOnlyHardCodedUpwindCellBasedRunner(args...)
    backend = CUDABackend()
    kernel_DivOnlyHardCodedUpwindCellBased(backend, 256)(args...; ndrange=length(args[1]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_DivOnlyHardCodedUpwindCellBased(
    @Const(cells),      # CuArray{GpuCell}
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
)
    iElement = @index(Global)
    cell = cells[iElement]
    numFaces = length(cell.iFaces)
    diag = 0.0
    nCells = length(cells)

    for iFace in 1:cell.nInternalFaces
        iFaceIndex = cell.iFaces[iFace]
        theFace = faces[iFaceIndex]

        diffusion = nus[iElement] * theFace.gDiff

        # Convection
        Uf = 0.5(U[iElement] + U[cell.iNeighbors[iFace]])                  # interpolate velocity to face 
        ϕf = dot(Uf, theFace.Sf)                    # flux through the face
        weights_f = upwind(ϕf)                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = ϕf * weights_f + diffusion
        valueLower = -ϕf * (1 - weights_f) - diffusion

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
        convection = bFaceValues[relativeFaceIndex] .* dot(theFace.Sf, bFaceValues[relativeFaceIndex])
        diffusion = nus[iElement] * theFace.gDiff

        diag -= diffusion
        RHS[iElement] -= diffusion - convection[1]
        RHS[iElement+nCells] -= diffusion - convection[2]
        RHS[iElement+nCells+nCells] -= diffusion - convection[3]
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end

function DivOnlyHardCodedCDFCellBasedRunner(args...)
    backend = CUDABackend()
    kernel_DivOnlyHardCodedCDFCellBased(backend, 256)(args...; ndrange=length(args[1]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_DivOnlyHardCodedCDFCellBased(
    @Const(cells),      # CuArray{GpuCell}
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
)
    iElement = @index(Global)
    cell = cells[iElement]
    numFaces = length(cell.iFaces)
    diag = 0.0
    nCells = length(cells)

    for iFace in 1:cell.nInternalFaces
        iFaceIndex = cell.iFaces[iFace]
        theFace = faces[iFaceIndex]

        diffusion = nus[iElement] * theFace.gDiff

        # Convection
        Uf = 0.5(U[iElement] + U[cell.iNeighbors[iFace]])                  # interpolate velocity to face 
        ϕf = dot(Uf, theFace.Sf)                    # flux through the face
        weights_f = 0.5                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = ϕf * weights_f + diffusion
        valueLower = -ϕf * (1 - weights_f) - diffusion

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
        convection = bFaceValues[relativeFaceIndex] .* dot(theFace.Sf, bFaceValues[relativeFaceIndex])
        diffusion = nus[iElement] * theFace.gDiff

        diag -= diffusion
        RHS[iElement] -= diffusion - convection[1]
        RHS[iElement+nCells] -= diffusion - convection[2]
        RHS[iElement+nCells+nCells] -= diffusion - convection[3]
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end

function DivOnlyDynamicCellBasedRunner(args, f)
    backend = CUDABackend()
    kernel_DivOnlyDynamicCellBased(backend, 256)(args..., f; ndrange=length(args[1]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_DivOnlyDynamicCellBased(
    @Const(cells),      # CuArray{GpuCell}
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
    @Const(divFunc)
)
    iElement = @index(Global)
    cell = cells[iElement]
    numFaces = length(cell.iFaces)
    diag = 0.0
    nCells = length(cells)

    for iFace in 1:cell.nInternalFaces
        iFaceIndex = cell.iFaces[iFace]
        theFace = faces[iFaceIndex]

        # Convection
        Uf = 0.5(U[iElement] + U[cell.iNeighbors[iFace]])                  # interpolate velocity to face 
        ϕf = dot(Uf, theFace.Sf)                    # flux through the face
        weights_f = divFunc(ϕf)                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)

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
        convection = bFaceValues[relativeFaceIndex] .* dot(theFace.Sf, bFaceValues[relativeFaceIndex])

        RHS[iElement] -= convection[1]
        RHS[iElement+nCells] -= convection[2]
        RHS[iElement+nCells+nCells] -= convection[3]
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end

function HardCodedUpwindCellBasedRunner(args...)
    backend = CUDABackend()
    kernel_HardCodedUpwindCellBased(backend, 256)(args...; ndrange=length(args[1]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_HardCodedUpwindCellBased(
    @Const(cells),      # CuArray{GpuCell}
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
)
    iElement = @index(Global)
    cell = cells[iElement]
    numFaces = length(cell.iFaces)
    diag = 0.0
    nCells = length(cells)

    for iFace in 1:cell.nInternalFaces
        iFaceIndex = cell.iFaces[iFace]
        theFace = faces[iFaceIndex]

        # Convection
        Uf = 0.5(U[iElement] + U[cell.iNeighbors[iFace]])                  # interpolate velocity to face 
        ϕf = dot(Uf, theFace.Sf)                    # flux through the face
        weights_f = upwind(ϕf)                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)

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
        convection = bFaceValues[relativeFaceIndex] .* dot(theFace.Sf, bFaceValues[relativeFaceIndex])

        RHS[iElement] -= convection[1]
        RHS[iElement+nCells] -= convection[2]
        RHS[iElement+nCells+nCells] -= convection[3]
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end

function HardCodedCDFCellBasedRunner(args...)
    backend = CUDABackend()
    kernel_HardCodedCDFCellBased(backend, 256)(args...; ndrange=length(args[1]))
    KernelAbstractions.synchronize(backend)
end

@kernel function kernel_HardCodedCDFCellBased(
    @Const(cells),      # CuArray{GpuCell}
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
)
    iElement = @index(Global)
    cell = cells[iElement]
    numFaces = length(cell.iFaces)
    diag = 0.0
    nCells = length(cells)

    for iFace in 1:cell.nInternalFaces
        iFaceIndex = cell.iFaces[iFace]
        theFace = faces[iFaceIndex]

        # Convection
        Uf = 0.5(U[iElement] + U[cell.iNeighbors[iFace]])                  # interpolate velocity to face 
        ϕf = dot(Uf, theFace.Sf)                    # flux through the face
        weights_f = centralDifferencing(ϕf)                              # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)

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
        convection = bFaceValues[relativeFaceIndex] .* dot(theFace.Sf, bFaceValues[relativeFaceIndex])

        RHS[iElement] -= convection[1]
        RHS[iElement+nCells] -= convection[2]
        RHS[iElement+nCells+nCells] -= convection[3]
    end
    idx = offsets[iElement]
    cols[idx] = iElement
    rows[idx] = iElement
    Atomix.@atomic vals[idx] += diag
end