include("init.jl")
include("cpu_helper.jl")

function DivOnlyPrecalculatedWeightsUpwindFaceBasedAssembly(input::MatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    weights = input.weightsUpwind
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = weights[iFace]                      # get precalculated weight

        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)

        vals[theFace.ownerIdx] += valueUpper
        vals[theFace.neighborRelNeighborIdx] += valueUpper
        vals[theFace.neighborIdx] += valueLower
        vals[theFace.ownerRelOwnerIdx] += valueLower
    end
    @inbounds for iBoundary in eachindex(mesh.boundaries)
        if velocity_boundary[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = mesh.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        @inbounds for iFace in startFace:endFace-1
            @inbounds theFace = mesh.faces[iFace]
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # RHS/Source
            @inbounds RHS[theFace.iOwner] -= convection[1]
            @inbounds RHS[theFace.iOwner+nCells] -= convection[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= convection[3]
        end
    end
    return rows, cols, vals, RHS
end

function DivOnlyPrecalculatedWeightsCDFFaceBasedAssembly(input::MatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    velocity_boundary = input.U_boundary
    weights = input.weightsCdf
    U = input.U_internal
    nCells = length(mesh.cells)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = weights[iFace]                      # get precalculated weight

        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        vals[theFace.ownerIdx] += valueUpper
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        vals[theFace.neighborRelNeighborIdx] += valueUpper

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        vals[theFace.neighborIdx] += valueLower
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        vals[theFace.ownerRelOwnerIdx] += valueLower
    end
    @inbounds for iBoundary in eachindex(mesh.boundaries)
        if velocity_boundary[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = mesh.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        @inbounds for iFace in startFace:endFace-1
            @inbounds theFace = mesh.faces[iFace]
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            # diffusion = nu[theFace.iOwner] * theFace.gDiff
            # setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            @inbounds RHS[theFace.iOwner] -= convection[1]
            @inbounds RHS[theFace.iOwner+nCells] -= convection[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= convection[3]
        end
    end
    return rows, cols, vals, RHS
end

function DivOnlyHardcodedUpwindFaceBasedAssembly(input::MatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = upwind(ϕf)                          # get weight of transport variable interpolation 
                                                        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = weights_f * ϕf
        valueLower = -ϕf * (1 - weights_f)
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        vals[theFace.ownerIdx] += valueUpper
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        vals[theFace.neighborRelNeighborIdx] += valueUpper

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        vals[theFace.neighborIdx] += valueLower
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        vals[theFace.ownerRelOwnerIdx] += valueLower
    end
    @inbounds for iBoundary in eachindex(mesh.boundaries)
        if velocity_boundary[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = mesh.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        @inbounds for iFace in startFace:endFace-1
            @inbounds theFace = mesh.faces[iFace]
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            # diffusion = nu[theFace.iOwner] * theFace.gDiff
            # setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            @inbounds RHS[theFace.iOwner] -= convection[1]
            @inbounds RHS[theFace.iOwner+nCells] -= convection[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= convection[3]
        end
    end
    return rows, cols, vals, RHS
end

function DivOnlyHardcodedCDFFaceBasedAssembly(input::MatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = 0.5                                 # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = weights_f * ϕf
        valueLower = -ϕf * (1 - weights_f)
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        vals[theFace.ownerIdx] += valueUpper
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        vals[theFace.neighborRelNeighborIdx] += valueUpper

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        vals[theFace.neighborIdx] += valueLower
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        vals[theFace.ownerRelOwnerIdx] += valueLower
    end
    @inbounds for iBoundary in eachindex(mesh.boundaries)
        if velocity_boundary[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = mesh.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        @inbounds for iFace in startFace:endFace-1
            @inbounds theFace = mesh.faces[iFace]
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            # diffusion = nu[theFace.iOwner] * theFace.gDiff
            # setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            @inbounds RHS[theFace.iOwner] -= convection[1]
            @inbounds RHS[theFace.iOwner+nCells] -= convection[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= convection[3]
        end
    end
    return rows, cols, vals, RHS
end

function DivOnlyDynamicCDFFaceBasedAssembly(input::MatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, div::Function) where {P<:AbstractFloat}
    mesh = input.mesh
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = div(ϕf)                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        vals[theFace.ownerIdx] += valueUpper
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        vals[theFace.neighborRelNeighborIdx] += valueUpper

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        vals[theFace.neighborIdx] += valueLower
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        vals[theFace.ownerRelOwnerIdx] += valueLower
    end
    @inbounds for iBoundary in eachindex(mesh.boundaries)
        if velocity_boundary[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = mesh.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        @inbounds for iFace in startFace:endFace-1
            @inbounds theFace = mesh.faces[iFace]
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # RHS/Source
            @inbounds RHS[theFace.iOwner] -= convection[1]
            @inbounds RHS[theFace.iOwner+nCells] -= convection[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= convection[3]
        end
    end
    return rows, cols, vals, RHS
end

function DivOnlyDynamicUpwindFaceBasedAssembly(input::MatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, div::Function) where {P<:AbstractFloat}
    mesh = input.mesh
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = div(ϕf)                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        vals[theFace.ownerIdx] += valueUpper
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        vals[theFace.neighborRelNeighborIdx] += valueUpper

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        vals[theFace.neighborIdx] += valueLower
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        vals[theFace.ownerRelOwnerIdx] += valueLower
    end
    @inbounds for iBoundary in eachindex(mesh.boundaries)
        if velocity_boundary[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = mesh.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        @inbounds for iFace in startFace:endFace-1
            @inbounds theFace = mesh.faces[iFace]
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            # diffusion = nu[theFace.iOwner] * theFace.gDiff
            # setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            @inbounds RHS[theFace.iOwner] -= convection[1]
            @inbounds RHS[theFace.iOwner+nCells] -= convection[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= convection[3]
        end
    end
    return rows, cols, vals, RHS
end

function LaplaceOnlyFaceBasedAssembly(input::MatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U_boundary
    nCells = length(mesh.cells)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor
        diffusion = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper = diffusion
        valueLower = -diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        vals[theFace.ownerIdx] += valueUpper
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        vals[theFace.neighborRelNeighborIdx] += valueUpper

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        vals[theFace.neighborIdx] += valueLower
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        vals[theFace.ownerRelOwnerIdx] += valueLower
    end
    @inbounds for iBoundary in eachindex(mesh.boundaries)
        if velocity_boundary[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = mesh.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        @inbounds for iFace in startFace:endFace-1
            @inbounds theFace = mesh.faces[iFace]
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            # ϕf  = theFace.Sf ⋅ U_b
            # @inbounds convection = U_b .* ϕf
            # diffusion 
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            @inbounds RHS[theFace.iOwner] -= diffusion[1]
            @inbounds RHS[theFace.iOwner+nCells] -= diffusion[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= diffusion[3]
        end
    end
    return rows, cols, vals, RHS
end

function PrecalculatedWeightsCDFFaceBasedAssembly(input::MatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    weights = input.weightsCdf
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = weights[iFace]                      # get precalculated weight

        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)
        diffusion = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper += diffusion
        valueLower -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        vals[theFace.ownerIdx] += valueUpper
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        vals[theFace.neighborRelNeighborIdx] += valueUpper

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        vals[theFace.neighborIdx] += valueLower
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        vals[theFace.ownerRelOwnerIdx] += valueLower
    end
    @inbounds for iBoundary in eachindex(mesh.boundaries)
        if velocity_boundary[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = mesh.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        @inbounds for iFace in startFace:endFace-1
            @inbounds theFace = mesh.faces[iFace]
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end

function PrecalculatedWeightsUpwindFaceBasedAssembly(input::MatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    weights = input.weightsUpwind
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = weights[iFace]                      # get precalculated weight

        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)
        diffusion = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper += diffusion
        valueLower -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        vals[theFace.ownerIdx] += valueUpper
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        vals[theFace.neighborRelNeighborIdx] += valueUpper

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        vals[theFace.neighborIdx] += valueLower
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        vals[theFace.ownerRelOwnerIdx] += valueLower
    end
    @inbounds for iBoundary in eachindex(mesh.boundaries)
        if velocity_boundary[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = mesh.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        @inbounds for iFace in startFace:endFace-1
            @inbounds theFace = mesh.faces[iFace]
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end

function HardcodedUpwindFaceBasedAssembly(input::MatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = upwind(ϕf)                          # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = weights_f * ϕf
        valueLower = -ϕf * (1 - weights_f)
        diffusion = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper += diffusion
        valueLower -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        vals[theFace.ownerIdx] += valueUpper
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        vals[theFace.neighborRelNeighborIdx] += valueUpper

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        vals[theFace.neighborIdx] += valueLower
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        vals[theFace.ownerRelOwnerIdx] += valueLower
    end
    @inbounds for iBoundary in eachindex(mesh.boundaries)
        if velocity_boundary[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = mesh.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        @inbounds for iFace in startFace:endFace-1
            @inbounds theFace = mesh.faces[iFace]
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end

function HardcodedCDFFaceBasedAssembly(input::MatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = 0.5                                 # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = weights_f * ϕf
        valueLower = -ϕf * (1 - weights_f)
        diffusion = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper += diffusion
        valueLower -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        vals[theFace.ownerIdx] += valueUpper
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        vals[theFace.neighborRelNeighborIdx] += valueUpper

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        vals[theFace.neighborIdx] += valueLower
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        vals[theFace.ownerRelOwnerIdx] += valueLower
    end
    @inbounds for iBoundary in eachindex(mesh.boundaries)
        if velocity_boundary[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = mesh.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        @inbounds for iFace in startFace:endFace-1
            @inbounds theFace = mesh.faces[iFace]
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end

function DynamicUpwindFaceBasedAssembly(input::MatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, div::Function) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = div(ϕf)                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)
        diffusion = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper += diffusion
        valueLower -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        vals[theFace.ownerIdx] += valueUpper
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        vals[theFace.neighborRelNeighborIdx] += valueUpper

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        vals[theFace.neighborIdx] += valueLower
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        vals[theFace.ownerRelOwnerIdx] += valueLower
    end
    @inbounds for iBoundary in eachindex(mesh.boundaries)
        if velocity_boundary[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = mesh.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        @inbounds for iFace in startFace:endFace-1
            @inbounds theFace = mesh.faces[iFace]
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end # function batchedFaceBasedAssembly

function DynamicCDFFaceBasedAssembly(input::MatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, div::Function) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = div(ϕf)                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper = ϕf * weights_f
        valueLower = -ϕf * (1 - weights_f)
        diffusion = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper += diffusion
        valueLower -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        vals[theFace.ownerIdx] += valueUpper
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        vals[theFace.neighborRelNeighborIdx] += valueUpper

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        vals[theFace.neighborIdx] += valueLower
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        vals[theFace.ownerRelOwnerIdx] += valueLower
    end
    @inbounds for iBoundary in eachindex(mesh.boundaries)
        if velocity_boundary[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = mesh.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        @inbounds for iFace in startFace:endFace-1
            @inbounds theFace = mesh.faces[iFace]
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end # function batchedFaceBasedAssembly