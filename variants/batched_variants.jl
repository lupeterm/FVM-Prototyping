include("../init.jl")
include("../cpu_helper.jl")
using Atomix

function FusedBatchedFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, batches, fused_pde::DiffEq) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    for id in batches.indices
        if id == -1
            continue
        end
        FusedBatch(batches[id], nu, fused_pde, vals, U, faces)
    end
    boundary_fused(input.boundaries, U_b, fused_pde, faces, nu, vals, RHS, nCells)
end

function FusedBatch(batch, nu, fused_pde, vals, U, faces)
    @batch for iFace in batch
        valueUpper, valueLower = fused_pde(U[faces.iOwner[iFace]], U[faces.iNeighbor[iFace]], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], 0.0, 0.0)

        vals[faces.ownerIdx[iFace]] += valueUpper
        vals[faces.neighborIdx[iFace]] += valueLower
        vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
        vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
    end
end

function PrecalculatedWeightsBatchedFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, batches, weights) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    for id in batches.indices
        if id == -1
            continue
        end
        PrecalculatedWeightsBatch(batches[id], nu, weights, vals, U, faces)
    end
    boundary(input.boundaries, U_b, faces, nu, vals, RHS, nCells)
end

function PrecalculatedWeightsBatch(batch, nu, weights, vals, U, faces)
    @batch for iFace in batch
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]

        Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
        ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
        weights_f = weights[iFace]                      # get precalculated weight

        valueUpper = ϕf * weights_f + diffusion
        valueLower = -ϕf * (1.0 - weights_f) - diffusion

        vals[faces.ownerIdx[iFace]] += valueUpper
        vals[faces.neighborIdx[iFace]] += valueLower
        vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
        vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
    end
end

function DynamicBatchedFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, batches, f_symbol::Symbol) where {P<:AbstractFloat}
    f = eval(f_symbol)
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    for id in batches.indices
        if id == -1
            continue
        end
        DynamicBatch(batches[id], nu, f, vals, U, faces)
    end
    boundary(input.boundaries, U_b, faces, nu, vals, RHS, nCells)
end

function DynamicBatch(batch, nu, divScheme, vals, U, faces)
    @batch for iFace in batch
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]

        Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
        ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
        weights_f = divScheme(ϕf)                      # get precalculated weight

        valueUpper = ϕf * weights_f + diffusion
        valueLower = -ϕf * (1.0 - weights_f) - diffusion

        vals[faces.ownerIdx[iFace]] += valueUpper
        vals[faces.neighborIdx[iFace]] += valueLower
        vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
        vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
    end
end

function HardCodedBatchedFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, batches, f_symbol::Symbol) where {P<:AbstractFloat}
    f = eval(f_symbol)
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    for id in batches.indices
        if id == -1
            continue
        end
        f(batches[id], nu, vals, U, faces)
    end
    boundary(input.boundaries, U_b, faces, nu, vals, RHS, nCells)
end
function HardCodedCDFBatch(batch, nu, vals, U, faces)
    @batch for iFace in batch
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]

        Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
        ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
        weights_f = cdf_f(ϕf)                      # get precalculated weight

        valueUpper = ϕf * weights_f + diffusion
        valueLower = -ϕf * (1.0 - weights_f) - diffusion

        vals[faces.ownerIdx[iFace]] += valueUpper
        vals[faces.neighborIdx[iFace]] += valueLower
        vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
        vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
    end
end

function HardCodedUpwindBatch(batch, nu, vals, U, faces)
    @batch for iFace in batch
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]

        Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
        ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
        weights_f = upwind_f(ϕf)                      # get precalculated weight

        valueUpper = ϕf * weights_f + diffusion
        valueLower = -ϕf * (1.0 - weights_f) - diffusion

        vals[faces.ownerIdx[iFace]] += valueUpper
        vals[faces.neighborIdx[iFace]] += valueLower
        vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
        vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
    end
end

function boundary(boundaries, U_b, faces, nu, vals, RHS, nCells)
    @batch for iBoundary in eachindex(boundaries)
        if U_b[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        for iFace in startFace:endFace-1
            @inbounds relativeFaceIndex = iFace - boundaries[iBoundary].startFace
            ϕf = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
            convection = U_b[iBoundary].values[relativeFaceIndex] .* ϕf

            diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
            # RHS/Source
            value = convection .+ diffusion
            @inbounds vals[faces.ownerIdx[iFace]] -= diffusion
            @inbounds RHS[faces.iOwner[iFace]] += value[1]
            @inbounds RHS[faces.iOwner[iFace]+nCells] += value[2]
            @inbounds RHS[faces.iOwner[iFace]+nCells+nCells] += value[3]
        end
    end
end


function boundary_fused(boundaries, U_b, fused_pde, faces, nu, vals, RHS, nCells)
    @batch for iBoundary in eachindex(boundaries)
        if U_b[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        for iFace in startFace:endFace-1
            @inbounds relativeFaceIndex = iFace - boundaries[iBoundary].startFace
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], 0.0, 0.0, 0.0, 0.0)

            Atomix.@atomic vals[faces.ownerIdx[iFace]] += diag
            # RHS/Source
            Atomix.@atomic RHS[faces.iOwner[iFace]] += rhsx
            Atomix.@atomic RHS[faces.iOwner[iFace]+nCells] += rhsy
            Atomix.@atomic RHS[faces.iOwner[iFace]+nCells+nCells] += rhsz
        end
    end
end
