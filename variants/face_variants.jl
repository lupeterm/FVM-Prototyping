include("../init.jl")
include("../cpu_helper.jl")
using Atomix


function FusedFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, fused_pde::DiffEq) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    for iFace in input.numInteriorFaces
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]

        valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], faces.Sf[iFace], nu[iOwner], faces.gDiff[iFace], zero(P), zero(P))

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

            diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
            # RHS/Source
            @inbounds RHS[faces.iOwner[iFace]] += rhsx
            @inbounds RHS[faces.iOwner[iFace]+nCells] += rhsy
            @inbounds RHS[faces.iOwner[iFace]+nCells+nCells] += rhsz
        end
    end
    return vals, RHS
end


function PrecalculatedWeightsFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, weights::Vector{P}) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    for iFace in input.numInteriorFaces
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]

        Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
        ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
        weights_f = weights[iFace]                      # get precalculated weight

        valueUpper::P = ϕf * weights_f + diffusion
        valueLower::P = -ϕf * (1.0 - weights_f) - diffusion

        vals[faces.ownerIdx[iFace]] += valueUpper
        vals[faces.neighborIdx[iFace]] += valueLower
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
            ϕf::P = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
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
    return  vals, RHS
end

function HardcodedUpwindFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    for iFace in input.numInteriorFaces
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]

        Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
        ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
        weights_f = upwind_f(ϕf)                      # get precalculated weight

        valueUpper::P = ϕf * weights_f + diffusion
        valueLower::P = -ϕf * (1.0 - weights_f) - diffusion

        vals[faces.ownerIdx[iFace]] += valueUpper
        vals[faces.neighborIdx[iFace]] += valueLower
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
            ϕf::P = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
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
    return  vals, RHS
end

function HardcodedCDFFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    for iFace in input.numInteriorFaces
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]

        Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
        ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
        weights_f = cdf_f(ϕf)                      # get precalculated weight

        valueUpper::P = ϕf * weights_f + diffusion
        valueLower::P = -ϕf * (1.0 - weights_f) - diffusion

        vals[faces.ownerIdx[iFace]] += valueUpper
        vals[faces.neighborIdx[iFace]] += valueLower
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
            ϕf::P = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
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
    return  vals, RHS
end

function DynamicFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, divScheme::Function) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    for iFace in input.numInteriorFaces
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]

        Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
        ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
        weights_f = divScheme(ϕf)                      # get precalculated weight

        valueUpper::P = ϕf * weights_f + diffusion
        valueLower::P = -ϕf * (1.0 - weights_f) - diffusion

        vals[faces.ownerIdx[iFace]] += valueUpper
        vals[faces.neighborIdx[iFace]] += valueLower
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
            ϕf::P = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
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
    return  vals, RHS
end

################
################ Serial
################
################
################
################ Parallel
################



function FusedFaceBasedAssembly_t(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, fused_pde::DiffEq) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    Threads.@threads for iFace in input.numInteriorFaces
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]

        valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], faces.Sf[iFace], nu[iOwner], faces.gDiff[iFace], zero(P), zero(P))

        Atomix.@atomic vals[faces.ownerIdx[iFace]] += valueUpper
        Atomix.@atomic vals[faces.neighborIdx[iFace]] += valueLower
        @inbounds vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
        @inbounds vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
    end
    @threads for iBoundary in eachindex(input.boundaries)
        if U_b[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = input.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        for iFace in startFace:endFace-1
            @inbounds relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            ϕf::P = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
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
    return vals, RHS
end



function PrecalculatedWeightsFaceBasedAssembly_t(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, weights::Vector{P}) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    Threads.@threads for iFace in input.numInteriorFaces
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]

        Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
        ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
        weights_f = weights[iFace]                      # get precalculated weight

        valueUpper::P = ϕf * weights_f + diffusion
        valueLower::P = -ϕf * (1.0 - weights_f) - diffusion

        Atomix.@atomic vals[faces.ownerIdx[iFace]] += valueUpper
        Atomix.@atomic vals[faces.neighborIdx[iFace]] += valueLower
        @inbounds vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
        @inbounds vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
    end
    @threads for iBoundary in eachindex(input.boundaries)
        if U_b[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = input.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        for iFace in startFace:endFace-1
            @inbounds relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            ϕf::P = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
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
    return  vals, RHS
end

function HardcodedUpwindFaceBasedAssembly_t(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    Threads.@threads for iFace in input.numInteriorFaces
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]

        Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
        ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
        weights_f = upwind_f(ϕf)                      # get precalculated weight

        valueUpper::P = ϕf * weights_f + diffusion
        valueLower::P = -ϕf * (1.0 - weights_f) - diffusion

        Atomix.@atomic vals[faces.ownerIdx[iFace]] += valueUpper
        Atomix.@atomic vals[faces.neighborIdx[iFace]] += valueLower
        @inbounds vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
        @inbounds vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
    end
    @threads for iBoundary in eachindex(input.boundaries)
        if U_b[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = input.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        for iFace in startFace:endFace-1
            @inbounds relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            ϕf::P = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
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
    return  vals, RHS
end

function HardcodedCDFFaceBasedAssembly_t(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    Threads.@threads for iFace in input.numInteriorFaces
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]

        Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
        ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
        weights_f = cdf_f(ϕf)                      # get precalculated weight

        valueUpper::P = ϕf * weights_f + diffusion
        valueLower::P = -ϕf * (1.0 - weights_f) - diffusion

        Atomix.@atomic vals[faces.ownerIdx[iFace]] += valueUpper
        Atomix.@atomic vals[faces.neighborIdx[iFace]] += valueLower
        @inbounds vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
        @inbounds vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
    end
    @threads for iBoundary in eachindex(input.boundaries)
        if U_b[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = input.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        for iFace in startFace:endFace-1
            @inbounds relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            ϕf::P = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
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
    return  vals, RHS
end

function DynamicFaceBasedAssembly_t(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, divScheme::Function) where {P<:AbstractFloat}
    nu = input.nu
    faces = input.faces
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(input.cells.index)
    Threads.@threads for iFace in input.numInteriorFaces
        iOwner = faces.iOwner[iFace]
        iNeighbor = faces.iNeighbor[iFace]
        diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]

        Uf = 0.5(U[iOwner] + U[iNeighbor])                             # interpolate velocity to face 
        ϕf = Uf ⋅ faces.Sf[iFace]                   # flux through the face
        weights_f = divScheme(ϕf)                      # get precalculated weight

        valueUpper::P = ϕf * weights_f + diffusion
        valueLower::P = -ϕf * (1.0 - weights_f) - diffusion

        Atomix.@atomic vals[faces.ownerIdx[iFace]] += valueUpper
        Atomix.@atomic vals[faces.neighborIdx[iFace]] += valueLower
        @inbounds vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
        @inbounds vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
    end
    @threads for iBoundary in eachindex(input.boundaries)
        if U_b[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = input.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        for iFace in startFace:endFace-1
            @inbounds relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
            ϕf::P = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
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
    return  vals, RHS
end
