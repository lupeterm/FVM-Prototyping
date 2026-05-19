include("../init.jl")
include("../cpu_helper.jl")
using Atomix


function FusedGlobalFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, fused_pde::DiffEq) where {P<:AbstractFloat}
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


function PrecalculatedWeightsGlobalFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, weights::Vector{P}) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    for iFace in eachindex(faces.iOwner)
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
        if faces.iNeighbor[iFace] > 0
            Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
            ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
            weights_f = weights[iFace]                      # get precalculated weight

            valueUpper::P = ϕf * weights_f + diffusion
            valueLower::P = -ϕf * (1.0 - weights_f) - diffusion

            vals[faces.ownerIdx[iFace]] += valueUpper
            vals[faces.neighborIdx[iFace]] += valueLower
            @inbounds vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
            @inbounds vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
        else
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            # convection
            ϕf::P = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
            convection = U_b[iBoundary].values[relativeFaceIndex] .* ϕf
            # RHS/Source
            value = convection .+ diffusion
            @inbounds vals[faces.ownerIdx[iFace]] -= diffusion
            @inbounds RHS[faces.iOwner[iFace]] += value[1]
            @inbounds RHS[faces.iOwner[iFace]+nCells] += value[2]
            @inbounds RHS[faces.iOwner[iFace]+nCells+nCells] += value[3]
        end
    end
    return vals, RHS
end

function HardcodedUpwindGlobalFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    for iFace in eachindex(faces.iOwner)
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
        if faces.iNeighbor[iFace] > 0
            Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
            ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
            weights_f = upwind_f(ϕf)                      # get precalculated weight

            valueUpper::P = ϕf * weights_f + diffusion
            valueLower::P = -ϕf * (1.0 - weights_f) - diffusion

            vals[faces.ownerIdx[iFace]] += valueUpper
            vals[faces.neighborIdx[iFace]] += valueLower
            @inbounds vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
            @inbounds vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
        else
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            # convection
            ϕf::P = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
            convection = U_b[iBoundary].values[relativeFaceIndex] .* ϕf
            # RHS/Source
            value = convection .+ diffusion
            @inbounds vals[faces.ownerIdx[iFace]] -= diffusion
            @inbounds RHS[faces.iOwner[iFace]] += value[1]
            @inbounds RHS[faces.iOwner[iFace]+nCells] += value[2]
            @inbounds RHS[faces.iOwner[iFace]+nCells+nCells] += value[3]
        end
    end
    return vals, RHS
end

function HardcodedCDFGlobalFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    for iFace in eachindex(faces.iOwner)
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
        if faces.iNeighbor[iFace] > 0
            Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
            ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
            weights_f = cdf_f(ϕf)                      # get precalculated weight

            valueUpper::P = ϕf * weights_f + diffusion
            valueLower::P = -ϕf * (1.0 - weights_f) - diffusion

            vals[faces.ownerIdx[iFace]] += valueUpper
            vals[faces.neighborIdx[iFace]] += valueLower
            @inbounds vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
            @inbounds vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
        else
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            # convection
            ϕf::P = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
            convection = U_b[iBoundary].values[relativeFaceIndex] .* ϕf
            # RHS/Source
            value = convection .+ diffusion
            @inbounds vals[faces.ownerIdx[iFace]] -= diffusion
            @inbounds RHS[faces.iOwner[iFace]] += value[1]
            @inbounds RHS[faces.iOwner[iFace]+nCells] += value[2]
            @inbounds RHS[faces.iOwner[iFace]+nCells+nCells] += value[3]
        end
    end
    return vals, RHS
end

function DynamicGlobalFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, divScheme::Function) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    for iFace in eachindex(faces.iOwner)
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
        if faces.iNeighbor[iFace] > 0
            Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
            ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
            weights_f = divScheme(ϕf)                      # get precalculated weight

            valueUpper::P = ϕf * weights_f + diffusion
            valueLower::P = -ϕf * (1.0 - weights_f) - diffusion

            vals[faces.ownerIdx[iFace]] += valueUpper
            vals[faces.neighborIdx[iFace]] += valueLower
            @inbounds vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
            @inbounds vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
        else
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            # convection
            ϕf::P = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
            convection = U_b[iBoundary].values[relativeFaceIndex] .* ϕf
            # RHS/Source
            value = convection .+ diffusion
            @inbounds vals[faces.ownerIdx[iFace]] -= diffusion
            @inbounds RHS[faces.iOwner[iFace]] += value[1]
            @inbounds RHS[faces.iOwner[iFace]+nCells] += value[2]
            @inbounds RHS[faces.iOwner[iFace]+nCells+nCells] += value[3]
        end
    end
    return vals, RHS
end

################
################ Serial
################
################
################
################ Parallel
################



function FusedGlobalFaceBasedAssembly_t(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, fused_pde::DiffEq) where {P<:AbstractFloat}
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



function PrecalculatedWeightsGlobalFaceBasedAssembly_t(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, weights::Vector{P}) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    Threads.@threads for iFace in eachindex(faces.iOwner)
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
        if faces.iNeighbor[iFace] > 0
            Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
            ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
            weights_f = weights[iFace]                      # get precalculated weight

            valueUpper::P = ϕf * weights_f + diffusion
            valueLower::P = -ϕf * (1.0 - weights_f) - diffusion

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
            # convection
            ϕf::P = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
            convection = U_b[iBoundary].values[relativeFaceIndex] .* ϕf
            # RHS/Source
            value = convection .+ diffusion
            @inbounds vals[faces.ownerIdx[iFace]] -= diffusion
            @inbounds RHS[faces.iOwner[iFace]] += value[1]
            @inbounds RHS[faces.iOwner[iFace]+nCells] += value[2]
            @inbounds RHS[faces.iOwner[iFace]+nCells+nCells] += value[3]
        end
    end
    return vals, RHS
end

function HardcodedUpwindGlobalFaceBasedAssembly_t(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    Threads.@threads for iFace in eachindex(faces.iOwner)
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
        if faces.iNeighbor[iFace] > 0
            Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
            ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
            weights_f = upwind_f(ϕf)                      # get precalculated weight

            valueUpper::P = ϕf * weights_f + diffusion
            valueLower::P = -ϕf * (1.0 - weights_f) - diffusion

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
            # convection
            ϕf::P = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
            convection = U_b[iBoundary].values[relativeFaceIndex] .* ϕf
            # RHS/Source
            value = convection .+ diffusion
            @inbounds vals[faces.ownerIdx[iFace]] -= diffusion
            @inbounds RHS[faces.iOwner[iFace]] += value[1]
            @inbounds RHS[faces.iOwner[iFace]+nCells] += value[2]
            @inbounds RHS[faces.iOwner[iFace]+nCells+nCells] += value[3]
        end
    end
    return vals, RHS
end

function HardcodedCDFGlobalFaceBasedAssembly_t(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    Threads.@threads for iFace in eachindex(faces.iOwner)
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
        if faces.iNeighbor[iFace] > 0
            Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
            ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
            weights_f = cdf_f(ϕf)                      # get precalculated weight

            valueUpper::P = ϕf * weights_f + diffusion
            valueLower::P = -ϕf * (1.0 - weights_f) - diffusion

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
            # convection
            ϕf::P = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
            convection = U_b[iBoundary].values[relativeFaceIndex] .* ϕf
            # RHS/Source
            value = convection .+ diffusion
            @inbounds vals[faces.ownerIdx[iFace]] -= diffusion
            @inbounds RHS[faces.iOwner[iFace]] += value[1]
            @inbounds RHS[faces.iOwner[iFace]+nCells] += value[2]
            @inbounds RHS[faces.iOwner[iFace]+nCells+nCells] += value[3]
        end
    end
    return vals, RHS
end

function DynamicGlobalFaceBasedAssembly_t(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, divScheme::Function) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    Threads.@threads for iFace in eachindex(faces.iOwner)
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
        if faces.iNeighbor[iFace] > 0
            Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
            ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
            weights_f = divScheme(ϕf)                      # get precalculated weight

            valueUpper::P = ϕf * weights_f + diffusion
            valueLower::P = -ϕf * (1.0 - weights_f) - diffusion

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
            # convection
            ϕf::P = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
            convection = U_b[iBoundary].values[relativeFaceIndex] .* ϕf
            # RHS/Source
            value = convection .+ diffusion
            @inbounds vals[faces.ownerIdx[iFace]] -= diffusion
            @inbounds RHS[faces.iOwner[iFace]] += value[1]
            @inbounds RHS[faces.iOwner[iFace]+nCells] += value[2]
            @inbounds RHS[faces.iOwner[iFace]+nCells+nCells] += value[3]
        end
    end
    return vals, RHS
end
