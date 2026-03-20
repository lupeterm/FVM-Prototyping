include("init.jl")
include("cpu_helper.jl")
include("operators.jl")


function prepf(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(P, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    return rows, vals, cols, RHS, offsets
end
function FusedFaceBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}, fused_pde::DiffEq) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor
        valueUpper, valueLower = zero(P), zero(P)
        valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], theFace.Sf, nu[iFace], theFace.gDiff, valueUpper, valueLower)

        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner])
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor)
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor])
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner)
    end
    @inbounds for iBoundary in eachindex(mesh.boundaries)
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

            setIndex!(theFace.iOwner, theFace.iOwner, diag, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            @inbounds RHS[theFace.iOwner] += rhsx
            @inbounds RHS[theFace.iOwner+nCells] += rhsy
            @inbounds RHS[theFace.iOwner+nCells+nCells] += rhsz
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