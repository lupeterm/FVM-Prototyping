include("init.jl")

function prepareRelativeIndices!(input::MatrixAssemblyInput)
    mesh = input.mesh
    cells = mesh.cells
    for cell in cells
        ownerIdx = -1
        for iFace in 1:cell.nInternalFaces
            iFaceIndex = cell.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            if ownerIdx == -1 && theFace.iOwner == cell.index && theFace.iOwner < theFace.iNeighbor
                theFace.relativeToOwner = 0
                ownerIdx = iFace
                continue
            end
            if cell.index == theFace.iOwner
                theFace.relativeToOwner = iFace - ownerIdx + 1
            else
                theFace.relativeToNeighbor = iFace - ownerIdx
            end
        end
        if ownerIdx == -1
            ownerIdx = cell.nInternalFaces + 1
        end
        for iFace in 1:cell.nInternalFaces
            iFaceIndex = cell.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            if cell.index == theFace.iOwner
                theFace.relativeToOwner = iFace - ownerIdx + 1
            else
                theFace.relativeToNeighbor = iFace - ownerIdx
            end
        end
    end
end

function gpu_precalcOffsets(input::MatrixAssemblyInput)::CuArray{Int32}
    mesh = input.mesh
    nCells = length(mesh.cells)
    offsets = ones(Int32, nCells)
    gpu_offsets = CuArray{Int32}(undef, nCells)
    negOffsets::Vector{Int32} = zeros(Int32, nCells)
    for iElement in 2:nCells
        offsets[iElement] += offsets[iElement-1] + mesh.cells[iElement-1].nInternalFaces
    end
    for iElement in 1:nCells
        theElement = mesh.cells[iElement]
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor > iElement
                negOffsets[theFace.iNeighbor] += 1
            end
        end
        offsets[iElement] += negOffsets[iElement]  # increase offset
    end
    copyto!(gpu_offsets, offsets)
    offsets = []
    negOffsets = []
    return gpu_offsets
end

function gpu_prepareValues(input::MatrixAssemblyInput)::CuArray{Float32}
    mesh = input.mesh
    nCells = length(mesh.cells)
    velocity_internal = input.U[2].values
    entries = nCells + 2 * mesh.numInteriorFaces
    vals_g = CuArray{Float32}(undef, entries * 3)
    vals_g[1:nCells] = getindex.(velocity_internal, 1)
    vals_g[entries+1:entries+nCells] = getindex.(velocity_internal, 2)
    vals_g[entries+entries+1:entries+entries+nCells] = getindex.(velocity_internal, 3)
    return vals_g
end

function gpu_prepareFaceBased(input::MatrixAssemblyInput)::Tuple
    mesh = input.mesh
    nu = input.nu
    nu_g = CuArray{Float32}(undef, nu.size[1])
    copyto!(nu_g, nu)
    nCells::Int32 = length(mesh.cells)
    RHS = CUDA.zeros(Float32, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = gpu_precalcOffsets(input)
    vals = gpu_prepareValues(input)
    rows::CuArray{Int32} = CUDA.zeros(Int32, entriesNeeded)
    cols = CUDA.zeros(Int32, entriesNeeded)
    N = mesh.numInteriorFaces
    threads::Int32 = 256
    internal_blocks::Int32 = cld(N, threads)
    prepareRelativeIndices!(input)
    M = mesh.numBoundaryFaces
    bblocks = cld(M, threads)
    bFaceValues = gpu_getBoundaryFaceValues(input)
    iOwners, iNeighbors, gDiffs, relativeToOwners, relativeToNbs = facesToGPUarrays(mesh.faces)
    return iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs, internal_blocks, bblocks, bFaceValues, RHS, nCells, M
end

function gpu_getBoundaryFaceValues(input::MatrixAssemblyInput)
    velocities = [b.values for b in input.U[1]]
    bFaceValues = zeros(Float32, (length(input.mesh.faces) - input.mesh.numInteriorFaces) * 3)
    i = 1
    for bvelocities in velocities
        for fVelocity in bvelocities
            bFaceValues[i:i+2] = fVelocity
            i += 3
        end
    end
    return CuArray(bFaceValues)
end

function gpu_estimate_data_facebased(input::MatrixAssemblyInput)
    mesh = input.mesh
    velocity_internal::Vector{MVector{3,Float32}} = input.U[2].values
    entries = length(mesh.cells) + 2 * mesh.numInteriorFaces
    nCells = length(mesh.cells)
    vals_g = CuArray{Float32}(undef, entries * 3)

    offsets::Vector{Int32} = ones(Int32, nCells)
    gpu_offsets = CuArray{Int32}(undef, nCells)
    for iElement in 2:nCells
        offsets[iElement] += offsets[iElement-1] + mesh.cells[iElement-1].nInternalFaces
    end
    vals_g[1:nCells] = getindex.(velocity_internal, 1)
    vals_g[entries+1:entries+nCells] = getindex.(velocity_internal, 2)
    vals_g[entries+entries+1:entries+entries+nCells] = getindex.(velocity_internal, 3)
    copyto!(gpu_offsets, offsets)
    offsets = []
    return gpu_offsets, vals_g
end

function facesToGPUarrays(faces)::Tuple{CuArray{Int32},CuArray{Int32},CuArray{Float32},CuArray{Int32},CuArray{Int32}}
    numFaces = length(faces)
    iOwners = CuArray{Int32}(undef, numFaces)
    iNeighbors = CuArray{Int32}(undef, numFaces)
    gDiffs = CuArray{Float32}(undef, numFaces)
    relativesO = CuArray{Int32}(undef, numFaces)
    relativesN = CuArray{Int32}(undef, numFaces)

    relativesN[1:numFaces] = [f.relativeToNeighbor for f in faces]
    relativesO[1:numFaces] = [f.relativeToOwner for f in faces]
    iOwners[1:numFaces] = [f.iOwner for f in faces]
    iNeighbors[1:numFaces] = [f.iNeighbor for f in faces]
    gDiffs[1:numFaces] = [f.gDiff for f in faces]
    return iOwners, iNeighbors, gDiffs, relativesO, relativesN
end