include("init.jl")
include("cpu_helper.jl")
using Atomix
function ThreadedCellBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}) where {P<:AbstractFloat}
    mesh = input.mesh
    nCells = length(mesh.cells)
    RHS = zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = getOffsets(input)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    vals = Vector{P}(undef, entriesNeeded * 3)
    chunks = Iterators.partition(1:nCells, nCells ÷ nthreads())
    chunks = [c for c in chunks]
    @threads for chunk in chunks
        CellBasedHelper(chunk, input, RHS, rows, cols, offsets, vals)
    end
    return rows, cols, vals
end

function CellBasedHelper(chunk::UnitRange, input::MatrixAssemblyInput, RHS::Vector{P}, rows::Vector{Int32}, cols::Vector{Int32}, offsets, vals::Vector) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    nCells = length(mesh.cells)
    velocity_boundary = input.U[1]
    velocity_internal = input.U[2].values
    for iElement in chunk
        theElement = mesh.cells[iElement]
        @inbounds diagx = velocity_internal[iElement][1]
        @inbounds diagy = velocity_internal[iElement][2]
        @inbounds diagz = velocity_internal[iElement][3]
        @inbounds for iFace in 1:theElement.nInternalFaces
            @inbounds iFaceIndex = theElement.iFaces[iFace]
            @inbounds theFace = mesh.faces[iFaceIndex]
            fluxCn = nu * theFace.gDiff
            fluxFn = -fluxCn
            idx = offsets[iElement] + iFace
            @inbounds cols[idx] = iElement
            @inbounds rows[idx] = theElement.iNeighbors[iFace]
            @inbounds vals[idx] = fluxFn    # x
            @inbounds vals[2*idx] = fluxFn  # y
            @inbounds vals[3*idx] = fluxFn  # z
            diagx += fluxCn
            diagy += fluxCn
            diagz += fluxCn
        end
        for iFace in theElement.nInternalFaces+1:length(theElement.iFaces)
            @inbounds iFaceIndex = theElement.iFaces[iFace]
            @inbounds theFace = mesh.faces[iFaceIndex]
            @inbounds iBoundary = mesh.faces[iFaceIndex].patchIndex
            @inbounds boundaryType = velocity_boundary[iBoundary].type
            if boundaryType == "fixedValue"
                fluxCb = nu * theFace.gDiff
                relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                fluxVb::Vector{P} = velocity_boundary[iBoundary].values[relativeFaceIndex] .* -fluxCb
                @inbounds RHS[iElement] -= fluxVb[1]
                @inbounds RHS[iElement+nCells] -= fluxVb[2]
                @inbounds RHS[iElement+nCells+nCells] -= fluxVb[3]
                diagx += fluxCb
                diagy += fluxCb
                diagz += fluxCb
            end
        end
        idx = offsets[iElement]
        @inbounds cols[idx] = iElement
        @inbounds rows[idx] = iElement
        @inbounds vals[idx] = diagx    # x
        @inbounds vals[2*idx] = diagy  # y
        @inbounds vals[3*idx] = diagz  # z
    end
end


function FusedFaceBasedAssemblyThreaded(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}, fused_pde::DiffEq) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    Threads.@threads for iFace in 1:mesh.numInteriorFaces
        theFace = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor
        valueUpper, valueLower = zero(P), zero(P)
        valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], theFace.Sf, nu[iOwner], theFace.gDiff, valueUpper, valueLower)

        idxUpper = offsets[iOwner]
        cols[idxUpper] = iOwner
        rows[idxUpper] = iOwner
        idx = offsets[iNeighbor]
        cols[idx] = iNeighbor
        rows[idx] = iNeighbor
        Atomix.@atomic vals[idxUpper] += valueUpper
        Atomix.@atomic vals[idx] += valueLower

        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        idx = offsets[iNeighbor] + theFace.relativeToNeighbor
        cols[idx] = iNeighbor
        rows[idx] = iOwner
        vals[idx] += valueUpper
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        idx = offsets[iOwner] + theFace.relativeToOwner
        cols[idx] = iOwner
        rows[idx] = iNeighbor
        vals[idx] += valueLower
    end
    Threads.@threads for iBoundary in eachindex(mesh.boundaries)
        if U_b[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = mesh.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        for iFace in startFace:endFace-1
            @inbounds theFace = mesh.faces[iFace]
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            diag, rhsx, rhsy, rhsz = zeros(P, 4)
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], theFace.Sf, nu[theFace.iOwner], theFace.gDiff, diag, rhsx, rhsy, rhsz)

            idx = offsets[theFace.iOwner]

            cols[idx] = theFace.iOwner
            rows[idx] = theFace.iOwner
            Atomix.@atomic vals[idx] += diag 
            # RHS/Source
            Atomix.@atomic RHS[theFace.iOwner] += rhsx
            Atomix.@atomic RHS[theFace.iOwner+nCells] += rhsy
            Atomix.@atomic RHS[theFace.iOwner+nCells+nCells] += rhsz
        end
    end
    return rows, cols, vals, RHS
end # function batchedFaceBasedAssembly

function FusedGlobalFaceBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}, fused_pde::DiffEq) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    for theFace in mesh.faces
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor
        if theFace.iNeighbor > 0
            valueUpper, valueLower = zero(P), zero(P)
            valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], theFace.Sf, nu[iOwner], theFace.gDiff, valueUpper, valueLower)

            # rowOwnStart + diagOffs[own]  -> (owner, owner)
            setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner])
            # rowNeiStart + neiOffs[facei] -> (neigh, owner)
            setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor)
            # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
            setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor])
            # rowOwnStart + ownOffs[facei] -> (owner, neigh)
            setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner)
        else
            iBoundary = theFace.patchIndex
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = theFace.index - mesh.boundaries[iBoundary].startFace
            # convection
            diag, rhsx, rhsy, rhsz = zeros(P, 4)
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], theFace.Sf, nu[theFace.iOwner], theFace.gDiff, diag, rhsx, rhsy, rhsz)

            setIndex!(theFace.iOwner, theFace.iOwner, diag, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            @inbounds RHS[theFace.iOwner] += rhsx
            @inbounds RHS[theFace.iOwner+nCells] += rhsy
            @inbounds RHS[theFace.iOwner+nCells+nCells] += rhsz
        end
    end

    return rows, cols, vals, RHS
end # function batchedFaceBasedAssembly

function FusedCellBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}, fused_pde::DiffEq) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    for theElement in mesh.cells
        iElement = theElement.index
        numFaces = length(theElement.iFaces)
        diag = 0.0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            valueUpper = 0.0
            valueLower = 0.0
            valueUpper, valueLower = fused_pde(U[iElement], U[theElement.iNeighbors[iFace]], theFace.Sf, nu[iFace], theFace.gDiff, valueUpper, valueLower)

            offdiag = ifelse(theFace.iOwner == iElement, valueLower, valueUpper)

            idx = offsets[iElement] + ifelse(theFace.iOwner == iElement, theFace.relativeToOwner, theFace.relativeToNeighbor)
            cols[idx] = iElement
            rows[idx] = theElement.iNeighbors[iFace]
            vals[idx] += offdiag

            diag += valueUpper
        end
        for iFace in theElement.nInternalFaces+1:numFaces
            iFaceIndex = theElement.iFaces[iFace]
            iBoundary = mesh.faces[iFaceIndex].patchIndex
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
            theFace = mesh.faces[iFaceIndex]
            rhsx, rhsy, rhsz = zeros(P, 3)
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], theFace.Sf, nu[theFace.iOwner], theFace.gDiff, diag, rhsx, rhsy, rhsz)
            RHS[iElement] += rhsx
            RHS[iElement+nCells] += rhsy
            RHS[iElement+nCells+nCells] += rhsz
        end
        setIndex!(theElement.index, theElement.index, diag, rows, cols, vals, offsets[iElement])
    end
    return rows, cols, vals, RHS
end # function CellBasedAssembly