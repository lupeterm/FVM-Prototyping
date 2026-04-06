include("init.jl")
include("cpu_helper.jl")


function FusedFaceBasedAssembly(input::MatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, fused_pde::DiffEq) where {P<:AbstractFloat}
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
        valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], theFace.Sf, nu[iOwner], theFace.gDiff, valueUpper, valueLower)

        vals[theFace.ownerIdx] += valueUpper
        vals[theFace.neighborIdx] += valueLower
        vals[theFace.neighborRelNeighborIdx] += valueUpper
        vals[theFace.ownerRelOwnerIdx] += valueLower
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

            vals[theFace.ownerIdx] += diag
            # RHS/Source
            @inbounds RHS[theFace.iOwner] += rhsx
            @inbounds RHS[theFace.iOwner+nCells] += rhsy
            @inbounds RHS[theFace.iOwner+nCells+nCells] += rhsz
        end
    end
    return vals, RHS
end # function batchedFaceBasedAssembly

function FusedGlobalFaceBasedAssembly(input::MatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, fused_pde::DiffEq) where {P<:AbstractFloat}
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

            vals[theFace.ownerIdx] += valueUpper
            vals[theFace.neighborIdx] += valueLower
            vals[theFace.neighborRelNeighborIdx] += valueUpper
            vals[theFace.ownerRelOwnerIdx] += valueLower
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

            vals[theFace.ownerIdx] += diag
            @inbounds RHS[theFace.iOwner] += rhsx
            @inbounds RHS[theFace.iOwner+nCells] += rhsy
            @inbounds RHS[theFace.iOwner+nCells+nCells] += rhsz
        end
    end
    return vals, RHS
end # function batchedFaceBasedAssembly

function FusedCellBasedAssembly(input::MatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, fused_pde::DiffEq) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    for theElement in mesh.cells
        iElement = theElement.index
        numFaces = length(theElement.iFaces)
        diag = zero(P)
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            valueUpper, valueLower = fused_pde(U[iElement], U[theElement.iNeighbors[iFace]], theFace.Sf, nu[iElement], theFace.gDiff, zero(P), zero(P))

            offdiag = ifelse(theFace.iOwner == iElement, valueLower, valueUpper)

            idx = ifelse(theFace.iOwner == iElement, theFace.neighborRelNeighborIdx, theFace.ownerRelOwnerIdx)
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
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], theFace.Sf, nu[theFace.iOwner], theFace.gDiff, diag, zero(P), zero(P), zero(P))
            RHS[iElement] += rhsx
            RHS[iElement+nCells] += rhsy
            RHS[iElement+nCells+nCells] += rhsz
        end
        vals[theElement.rowOffset] = diag
    end
    return vals, RHS
end # function CellBasedAssembly