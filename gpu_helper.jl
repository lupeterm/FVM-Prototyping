include("init.jl")


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

function gpu_prepareFaceBased(input::MatrixAssemblyInput{P})::Tuple where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    nu_g = CuArray{P}(undef, length(nu))
    copyto!(nu_g, nu)
    nCells::Int32 = length(mesh.cells)
    RHS = CUDA.zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = gpu_precalcOffsets(input)
    vals = CUDA.zeros(P, entriesNeeded)
    rows::CuArray{Int32} = CUDA.zeros(Int32, entriesNeeded)
    cols = CUDA.zeros(Int32, entriesNeeded)
    N::Int32 = length(mesh.faces)
    M = mesh.numBoundaryFaces
    threads::Int32 = 256
    numBlocks::Int32 = cld(N, threads)
    prepareRelativeIndices!(input)
    bFaceValues, U, bFaceMapping = gpu_getFaceValues(input)
    iOwners, iNeighbors, gDiffs, relativeToOwners, relativeToNbs, Sf = facesToGPUarrays(P, mesh.faces)
    return iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, mesh.numInteriorFaces, relativeToNbs, numBlocks, bFaceValues, RHS, nCells, M, U, Sf, bFaceMapping
end

function gpu_getFaceValues(input::MatrixAssemblyInput{P})::Tuple where {P<:AbstractFloat}
    useBfaces::Vector{Int32} = fill(-1, input.mesh.numBoundaryFaces)
    i = 1
    for iBoundary in eachindex(input.mesh.boundaries)
        theBoundary = input.mesh.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        for iFace in startFace:endFace-1
            theFace = input.mesh.faces[iFace]
            if input.U[1][iBoundary].type == "fixedValue"
                useBfaces[theFace.index - input.mesh.numInteriorFaces] = i
                i += 1
            end
        end
    end
    velocities = [b.values for b in input.U[1]]
    return CuArray(reduce(vcat, velocities)), CuArray(input.U[2].values), CuArray(useBfaces)
end

function gpu_estimate_data_facebased(input::MatrixAssemblyInput{P})::Tuple where {P<:AbstractFloat}
    mesh = input.mesh
    velocity_internal::Vector{MVector{3,P}} = input.U[2].values
    entries = length(mesh.cells) + 2 * mesh.numInteriorFaces
    nCells = length(mesh.cells)
    vals_g = CuArray{P}(undef, entries)

    offsets::Vector{Int32} = ones(Int32, nCells)
    gpu_offsets = CuArray{Int32}(undef, nCells)
    for iElement in 2:nCells
        offsets[iElement] += offsets[iElement-1] + mesh.cells[iElement-1].nInternalFaces
    end
    copyto!(gpu_offsets, offsets)
    offsets = []
    return gpu_offsets, vals_g
end

function facesToGPUarrays(P::Type{<:AbstractFloat}, faces::Vector{Face})
    numFaces = length(faces)
    iOwners = CuArray{Int32}(undef, numFaces)
    iNeighbors = CuArray{Int32}(undef, numFaces)
    gDiffs = CuArray{P}(undef, numFaces)
    relativesO = CuArray{Int32}(undef, numFaces)
    relativesN = CuArray{Int32}(undef, numFaces)
    Sf = CuArray{SVector{3,P}}(undef, numFaces)

    relativesN[1:numFaces] = [f.relativeToNeighbor for f in faces]
    relativesO[1:numFaces] = [f.relativeToOwner for f in faces]
    iOwners[1:numFaces] = [f.iOwner for f in faces]
    iNeighbors[1:numFaces] = [f.iNeighbor for f in faces]
    gDiffs[1:numFaces] = [f.gDiff for f in faces]
    Sf[1:numFaces] = [SVector(f.Sf) for f in faces]

    return iOwners, iNeighbors, gDiffs, relativesO, relativesN, Sf
end

# Idea was to precalculate corner and edge cells and only do atomic operations for those faces.
# these faces would be 2% for the LDC-M case
# ended up being slower than just using the atomics for all faces 
# function findCornerCells(input::MatrixAssemblyInput)
#     useAtomics = falses(input.mesh.numBoundaryFaces)
#     for cell::Cell in input.mesh.cells
#         if length(cell.iFaces) - cell.nInternalFaces > 1
#             for bFace in cell.nInternalFaces+1:length(cell.iFaces)
#                 useAtomics[cell.iFaces[bFace] - input.mesh.numInteriorFaces] = true
#             end
#         end
#     end
#     return CuArray(useAtomics)
# end