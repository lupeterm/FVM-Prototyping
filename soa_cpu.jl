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
        valueUpper, valueLower = zero(P), zero(P)
        @inbounds valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], faces.Sf[iFace], nu[iOwner], faces.gDiff[iFace], valueUpper, valueLower)

        Atomix.@atomic vals[faces.ownerIdx[iFace]] += valueUpper
        Atomix.@atomic vals[faces.neighborIdx[iFace]] += valueLower
        @inbounds vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
        @inbounds vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
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
            diag, rhsx, rhsy, rhsz = zeros(P, 4)
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], diag, rhsx, rhsy, rhsz)
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
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        valueUpper, valueLower = zero(P), zero(P)
        valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], faces.Sf[iFace], nu[iOwner], faces.gDiff[iFace], valueUpper, valueLower)

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
            diag, rhsx, rhsy, rhsz = zeros(P, 4)
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], diag, rhsx, rhsy, rhsz)

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
            valueUpper, valueLower = zero(P), zero(P)
            valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], faces.Sf[iFace], nu[iOwner], faces.gDiff[iFace], valueUpper, valueLower)

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
            diag, rhsx, rhsy, rhsz = zeros(P, 4)
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], diag, rhsx, rhsy, rhsz)

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
            valueUpper, valueLower = zero(P), zero(P)
            valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], faces.Sf[iFace], nu[iOwner], faces.gDiff[iFace], valueUpper, valueLower)

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
            diag, rhsx, rhsy, rhsz = zeros(P, 4)
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], diag, rhsx, rhsy, rhsz)

            @inbounds vals[faces.ownerIdx[iFace]] += diag
            # RHS/Source
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
        numFaces = length(cells.iFaces[iElement])
        diag = zero(P)
        for localIndex in 1:cells.nInternalFaces[iElement]
            iFace = cells.iFaces[iElement][localIndex]
            valueUpper = zero(P)
            valueLower = zero(P)
            valueUpper, valueLower = fused_pde(U[iElement], U[cells.iNeighbors[iElement][localIndex]], faces.Sf[iFace], nu[iElement], faces.gDiff[iFace], valueUpper, valueLower)

            offdiag = ifelse(faces.iOwner[iFace] == iElement, valueLower, valueUpper)

            idx = ifelse(faces.iOwner[iFace] == iElement, faces.ownerRelOwnerIdx[iFace], faces.neighborRelNeighborIdx[iFace])
            Atomix.@atomic vals[idx] += offdiag

            diag += valueUpper
        end
        for localIndex in cells.nInternalFaces[iElement]+1:numFaces
            iFace = cells.iFaces[iElement][localIndex]
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            rhsx, rhsy, rhsz = zeros(P, 3)
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[iElement], faces.gDiff[iFace], diag, rhsx, rhsy, rhsz)
            Atomix.@atomic RHS[iElement] += rhsx
            Atomix.@atomic RHS[iElement+nCells] += rhsy
            Atomix.@atomic RHS[iElement+nCells+nCells] += rhsz
        end
        idx = cells.rowOffset[iElement]
        vals[idx] += diag
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
        numFaces = length(cells.iFaces[iElement])
        diag = zero(P)
        for localIndex in 1:cells.nInternalFaces[iElement]
            iFace = cells.iFaces[iElement][localIndex]
            valueUpper = zero(P)
            valueLower = zero(P)
            valueUpper, valueLower = fused_pde(U[iElement], U[cells.iNeighbors[iElement][localIndex]], faces.Sf[iFace], nu[iElement], faces.gDiff[iFace], valueUpper, valueLower)

            offdiag = ifelse(faces.iOwner[iFace] == iElement, valueLower, valueUpper)

            idx = ifelse(faces.iOwner[iFace] == iElement, faces.ownerRelOwnerIdx[iFace], faces.neighborRelNeighborIdx[iFace])
            vals[idx] += offdiag

            diag += valueUpper
        end
        for localIndex in cells.nInternalFaces[iElement]+1:numFaces
            iFace = cells.iFaces[iElement][localIndex]
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            rhsx, rhsy, rhsz = zeros(P, 3)
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[iElement], faces.gDiff[iFace], diag, rhsx, rhsy, rhsz)
            RHS[iElement] += rhsx
            RHS[iElement+nCells] += rhsy
            RHS[iElement+nCells+nCells] += rhsz
        end
        idx = cells.rowOffset[iElement]
        vals[idx] += diag
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
            valueUpper, valueLower = zero(P), zero(P)
            valueUpper, valueLower = fused_pde(U[faces.iOwner[iFace]], U[faces.iNeighbor[iFace]], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], valueUpper, valueLower)

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
            diag, rhsx, rhsy, rhsz = zeros(P, 4)
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], diag, rhsx, rhsy, rhsz)

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
            valueUpper, valueLower = zero(P), zero(P)
            valueUpper, valueLower = fused_pde(U[faces.iOwner[iFace]], U[faces.iNeighbor[iFace]], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], valueUpper, valueLower)

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
            diag, rhsx, rhsy, rhsz = zeros(P, 4)
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], diag, rhsx, rhsy, rhsz)

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
    vals[ownerIdx] += valueUpper
    vals[neighborIdx] += valueLower
    vals[neighborRelNeighborIdx] += valueUpper
    vals[ownerRelOwnerIdx] += valueLower
end

# const SOAFUNCTIONS = [SOAFusedFaceBasedAssembly, SOAFusedFaceBasedAssemblyThreaded, SOAFusedCellBasedAssembly, SOAFusedCellBasedAssemblyThreaded, SOAFusedGlobalFaceBasedAssembly, SOAFusedGlobalFaceBasedAssemblyThreaded, SOAFusedBatchedFaceBasedAssemblyThreaded]
