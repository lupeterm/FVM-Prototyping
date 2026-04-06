include("init.jl")
include("cpu_helper.jl")
using Atomix
using SplitApplyCombine


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

        Atomix.@atomic vals[theFace.ownerIdx] += valueUpper
        Atomix.@atomic vals[theFace.neighborIdx] += valueLower

        vals[theFace.neighborRelNeighborIdx] += valueUpper
        vals[theFace.ownerRelOwnerIdx] += valueLower
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

function FusedGlobalFaceBasedAssemblyThreaded(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}, fused_pde::DiffEq) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    Threads.@threads for theFace in mesh.faces
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor
        if theFace.iNeighbor > 0
            valueUpper, valueLower = zero(P), zero(P)
            valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], theFace.Sf, nu[iOwner], theFace.gDiff, valueUpper, valueLower)

            idxUpper = offsets[iOwner]
            cols[idxUpper] = iOwner
            rows[idxUpper] = iOwner
            Atomix.@atomic vals[idxUpper] += valueUpper

            idx = offsets[iNeighbor]
            cols[idx] = iNeighbor
            rows[idx] = iNeighbor
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
        else
            iBoundary = theFace.patchIndex
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = theFace.index - mesh.boundaries[iBoundary].startFace
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

function FusedCellBasedAssemblyThreaded(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}, fused_pde::DiffEq) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    Threads.@threads for theElement in mesh.cells
        iElement = theElement.index
        numFaces = length(theElement.iFaces)
        diag = zero(P)
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            valueUpper = zero(P)
            valueLower = zero(P)
            valueUpper, valueLower = fused_pde(U[iElement], U[theElement.iNeighbors[iFace]], theFace.Sf, nu[iElement], theFace.gDiff, valueUpper, valueLower)

            offdiag = ifelse(theFace.iOwner == iElement, valueLower, valueUpper)

            idx = offsets[iElement] + ifelse(theFace.iOwner == iElement, theFace.relativeToOwner, theFace.relativeToNeighbor)
            cols[idx] = iElement
            rows[idx] = theElement.iNeighbors[iFace]
            Atomix.@atomic vals[idx] += offdiag

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
        idx = offsets[iElement]
        cols[idx] = iElement
        rows[idx] = iElement
        Atomix.@atomic vals[idx] += diag
    end
    return rows, cols, vals, RHS
end # function CellBasedAssembly

function greedyEdgeColoring(input::MatrixAssemblyInput)
    mesh = input.mesh
    for face::Face in mesh.faces
        face.batchId = -1
    end
    faceColorMapping = zeros(Int32, mesh.numInteriorFaces)
    for cell::Cell in mesh.cells
        usedColors = []
        for iFace in 1:cell.nInternalFaces
            iFaceIndex = cell.iFaces[iFace]
            face::Face = mesh.faces[iFaceIndex]
            if face.batchId == -1
                continue
            end
            push!(usedColors, face.batchId)
        end
        id = 1
        for iFace in 1:cell.nInternalFaces
            iFaceIndex = cell.iFaces[iFace]
            face::Face = mesh.faces[iFaceIndex]
            if face.batchId != -1
                continue
            end
            go = true
            while go
                if id in usedColors
                    id += 1
                    continue
                end
                face.batchId = id
                faceColorMapping[face.index] = id
                push!(usedColors, face.batchId)
                go = false
            end
        end
    end
end


function cpuBatchedPrep(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    groupedFaces = getBatches(input)
    prep = prepf(input)
    return input, prep..., groupedFaces
end

function getBatches(input::MatrixAssemblyInput)
    greedyEdgeColoring(input)
    return group(x -> x.batchId, x -> x.index, input.mesh.faces)
end

function FusedBatchedFaceBasedAssemblyThreaded(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}, batches, fused_pde::DiffEq) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    for id in batches.indices
        if id == -1
            continue
        end
        Threads.@threads for theFace in batches[id]
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
            vals[idxUpper] += valueUpper
            vals[idx] += valueLower

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
end