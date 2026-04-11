include("init.jl")
include("cpu_helper.jl")
using Atomix

function SOAFusedFaceBasedAssemblyThreaded(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, fused_pde::DiffEq) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    Threads.@threads for iFace in 1:input.numInteriorFaces
        @inbounds iOwner = faces.iOwner[iFace]
        @inbounds iNeighbor = faces.iNeighbor[iFace]
        @inbounds valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], faces.Sf[iFace], nu[iOwner], faces.gDiff[iFace], zero(P), zero(P))

        Atomix.@atomic vals[faces.ownerIdx[iFace]] += valueUpper
        Atomix.@atomic vals[faces.neighborIdx[iFace]] += valueLower
        @inbounds vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
        @inbounds vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
    end
    Threads.@threads for iBoundary in eachindex(input.boundaries)
        if U_b[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = input.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        for iFace in startFace:endFace-1
            @inbounds relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], zero(P), zero(P), zero(P), zero(P))
            Atomix.@atomic vals[faces.ownerIdx[iFace]] += diag
            # RHS/Source
            Atomix.@atomic RHS[faces.iOwner[iFace]] += rhsx
            Atomix.@atomic RHS[faces.iOwner[iFace]+nCells] += rhsy
            Atomix.@atomic RHS[faces.iOwner[iFace]+nCells+nCells] += rhsz
        end
    end
    return vals, RHS
end # function batchedFaceBasedAssembly


function SOAFusedFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, fused_pde::DiffEq) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    @inbounds for iFace in 1:input.numInteriorFaces
        @inbounds iOwner = faces.iOwner[iFace]
        @inbounds iNeighbor = faces.iNeighbor[iFace]
        @inbounds valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], faces.Sf[iFace], nu[iOwner], faces.gDiff[iFace], zero(P), zero(P))

        @inbounds vals[faces.ownerIdx[iFace]] += valueUpper
        @inbounds vals[faces.neighborIdx[iFace]] += valueLower
        @inbounds vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
        @inbounds vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
    end
    @inbounds for iBoundary in eachindex(input.boundaries)
        if U_b[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = input.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        for iFace in startFace:endFace-1
            @inbounds relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], zero(P), zero(P), zero(P), zero(P))

            @inbounds vals[faces.ownerIdx[iFace]] += diag
            # RHS/Source
            @inbounds RHS[faces.iOwner[iFace]] += rhsx
            @inbounds RHS[faces.iOwner[iFace]+nCells] += rhsy
            @inbounds RHS[faces.iOwner[iFace]+nCells+nCells] += rhsz
        end
    end
    return vals, RHS
end # function batchedFaceBasedAssembly

function SOAFusedGlobalFaceBasedAssemblyThreaded(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, fused_pde::DiffEq) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    Threads.@threads for iFace in eachindex(faces.iOwner)
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        if faces.iNeighbor[iFace] > 0
            valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], faces.Sf[iFace], nu[iOwner], faces.gDiff[iFace], zero(P), zero(P))

            Atomix.@atomic vals[faces.ownerIdx[iFace]] += valueUpper
            Atomix.@atomic vals[faces.neighborIdx[iFace]] += valueLower
            @inbounds vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
            @inbounds vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
        else
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], zero(P), zero(P), zero(P), zero(P))

            Atomix.@atomic vals[faces.ownerIdx[iFace]] += diag
            # RHS/Source
            Atomix.@atomic RHS[faces.iOwner[iFace]] += rhsx
            Atomix.@atomic RHS[faces.iOwner[iFace]+nCells] += rhsy
            Atomix.@atomic RHS[faces.iOwner[iFace]+nCells+nCells] += rhsz
        end
    end
    return vals, RHS

end

function SOAFusedGlobalFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, fused_pde::DiffEq) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    for iFace in eachindex(faces.iOwner)
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        if faces.iNeighbor[iFace] > 0
            valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], faces.Sf[iFace], nu[iOwner], faces.gDiff[iFace], zero(P), zero(P))

            @inbounds vals[faces.ownerIdx[iFace]] += valueUpper
            @inbounds vals[faces.neighborIdx[iFace]] += valueLower
            @inbounds vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
            @inbounds vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
        else
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], zero(P), zero(P), zero(P), zero(P))

            @inbounds vals[faces.ownerIdx[iFace]] += diag
            @inbounds RHS[faces.iOwner[iFace]] += rhsx
            @inbounds RHS[faces.iOwner[iFace]+nCells] += rhsy
            @inbounds RHS[faces.iOwner[iFace]+nCells+nCells] += rhsz
        end
    end
    return vals, RHS

end


function SOAFusedCellBasedAssemblyThreaded(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, fused_pde::DiffEq) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    cells = input.cells
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(cells.index)
    Threads.@threads for iElement in eachindex(cells.index)
        diag, rhsx, rhsy, rhsz = zero(P), zero(P), zero(P), zero(P)
        for localIndex in 1:cells.nInternalFaces[iElement]
            iFace = cells.iFaces[iElement][localIndex]
            isOwner = faces.iOwner[iFace] == iElement
            valueUpper, valueLower = fused_pde(U[iElement], U[faces.iNeighbor[iFace]], faces.Sf[iFace], nu[iElement], faces.gDiff[iFace], zero(P), zero(P))
            idx = ifelse(isOwner, faces.ownerRelOwnerIdx[iFace], faces.neighborRelNeighborIdx[iFace])
            vals[idx] += ifelse(isOwner, valueLower, valueUpper)
            diag += ifelse(!isOwner, valueLower, valueUpper)
        end
        for iFace in cells.iFaces[iElement][cells.nInternalFaces[iElement]+1:end]
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[iElement], faces.gDiff[iFace], diag, rhsx, rhsy, rhsz)
        end
        RHS[iElement] += rhsx
        RHS[iElement+nCells] += rhsy
        RHS[iElement+nCells+nCells] += rhsz
        vals[cells.rowOffset[iElement]] += diag
    end
    return vals, RHS
end
function SOAFusedCellBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, fused_pde::DiffEq) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    cells = input.cells
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(cells.index)
    for iElement in eachindex(cells.index)
        diag, rhsx, rhsy, rhsz = zero(P), zero(P), zero(P), zero(P)
        for localIndex in 1:cells.nInternalFaces[iElement]
            iFace = cells.iFaces[iElement][localIndex]
            isOwner = faces.iOwner[iFace] == iElement
            valueUpper, valueLower = fused_pde(U[iElement], U[faces.iNeighbor[iFace]], faces.Sf[iFace], nu[iElement], faces.gDiff[iFace], zero(P), zero(P))
            idx = ifelse(isOwner, faces.ownerRelOwnerIdx[iFace], faces.neighborRelNeighborIdx[iFace])
            vals[idx] += ifelse(isOwner, valueLower, valueUpper)
            diag += ifelse(!isOwner, valueLower, valueUpper)
        end
        for iFace in cells.iFaces[iElement][cells.nInternalFaces[iElement]+1:end]
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[iElement], faces.gDiff[iFace], diag, rhsx, rhsy, rhsz)
        end
        RHS[iElement] += rhsx
        RHS[iElement+nCells] += rhsy
        RHS[iElement+nCells+nCells] += rhsz
        vals[cells.rowOffset[iElement]] += diag
    end
    return vals, RHS
end

function SOAFusedBatchedFaceBasedAssemblyThreaded(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, batches, fused_pde::DiffEq) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    for id in batches.indices
        if id == -1
            continue
        end
        Threads.@threads for iFace in batches[id]
            valueUpper, valueLower = fused_pde(U[faces.iOwner[iFace]], U[faces.iNeighbor[iFace]], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], zero(P), zero(P))

            vals[faces.ownerIdx[iFace]] += valueUpper
            vals[faces.neighborIdx[iFace]] += valueLower
            vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
            vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
        end
    end
    Threads.@threads for iBoundary in eachindex(input.boundaries)
        if U_b[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = input.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        for iFace in startFace:endFace-1
            @inbounds relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], zero(P), zero(P), zero(P), zero(P))

            Atomix.@atomic vals[faces.ownerIdx[iFace]] += diag
            # RHS/Source
            Atomix.@atomic RHS[faces.iOwner[iFace]] += rhsx
            Atomix.@atomic RHS[faces.iOwner[iFace]+nCells] += rhsy
            Atomix.@atomic RHS[faces.iOwner[iFace]+nCells+nCells] += rhsz
        end
    end
end

function SOAFusedBatchedFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, batches, fused_pde::DiffEq) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    for id in batches.indices
        if id == -1
            continue
        end
        for iFace in batches[id]
            valueUpper, valueLower = 
            valueUpper, valueLower = fused_pde(U[faces.iOwner[iFace]], U[faces.iNeighbor[iFace]], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], zero(P), zero(P))

            vals[faces.ownerIdx[iFace]] += valueUpper
            vals[faces.neighborIdx[iFace]] += valueLower
            vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
            vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
        end
    end
    for iBoundary in eachindex(input.boundaries)
        if U_b[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = input.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        for iFace in startFace:endFace-1
            @inbounds relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], zero(P), zero(P), zero(P), zero(P))

            vals[faces.ownerIdx[iFace]] += diag
            # RHS/Source
            RHS[faces.iOwner[iFace]] += rhsx
            RHS[faces.iOwner[iFace]+nCells] += rhsy
            RHS[faces.iOwner[iFace]+nCells+nCells] += rhsz
        end
    end
end

# @inline innerFace(
#     fused_pde,
#     vals,
#     faces.ownerIdx[iFace],
#     faces.neighborIdx[iFace],
#     faces.neighborRelNeighborIdx[iFace],
#     faces.ownerRelOwnerIdx[iFace],
#     U[faces.iOwner[iFace]],
#     U[faces.iNeighbor[iFace]],
#     faces.Sf[iFace],
#     nu[faces.iOwner[iFace]],
#     faces.gDiff[iFace]
# )
function innerFace(
    fused_pde::DiffEq,
    vals::Vector{P},
    ownerIdx::Int32,
    neighborIdx::Int32,
    neighborRelNeighborIdx::Int32,
    ownerRelOwnerIdx::Int32,
    U_c::SVector{3,P},
    U_n::SVector{3,P},
    Sf::SVector{3,P},
    nu::P,
    gDiff::P
) where {P<:AbstractFloat}
    valueUpper, valueLower = fused_pde(U_c, U_n, Sf, nu, gDiff, zero(P), zero(P))
    Atomix.@atomic vals[ownerIdx] += valueUpper
    Atomix.@atomic vals[neighborIdx] += valueLower
    vals[neighborRelNeighborIdx] += valueUpper
    vals[ownerRelOwnerIdx] += valueLower
end
