include("init.jl")
include("cpu_helper.jl")

# function _FaceBasedAssembly(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
#     mesh = input.mesh
#     nu = input.nu
#     velocity_boundary = input.U[1]
#     nCells = length(mesh.cells)
#     entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
#     RHS = zeros(P, nCells * 3)
#     offsets = input.offsets
#     vals = zeros(P, entriesNeeded)
#     rows = zeros(Int32, entriesNeeded)
#     cols = zeros(Int32, entriesNeeded)


#     @inbounds for iFace in 1:mesh.numInteriorFaces
#         theFace = mesh.faces[iFace]
#         iOwner = theFace.iOwner
#         iNeighbor = theFace.iNeighbor
#         diffusion = nu[iOwner] * theFace.gDiff
#         fluxFn = -diffusion
#         # set diag and upper
#         setIndex!(iOwner, iOwner, diffusion, rows, cols, vals, offsets[iOwner])
#         setIndex!(iOwner, iNeighbor, fluxFn, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner)
#         # set diag and lower
#         setIndex!(iNeighbor, iNeighbor, diffusion, rows, cols, vals, offsets[iNeighbor])
#         setIndex!(iNeighbor, iOwner, fluxFn, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor)
#     end
#     @inbounds for iBoundary in eachindex(mesh.boundaries)
#         if velocity_boundary[iBoundary].type != "fixedValue"
#             continue
#         end
#         @inbounds theBoundary = mesh.boundaries[iBoundary]
#         startFace = theBoundary.startFace + 1
#         endFace = startFace + theBoundary.nFaces
#         @inbounds for iFace in startFace:endFace-1
#             @inbounds theFace = mesh.faces[iFace]
#             @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
#             # convection
#             U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
#             ϕf = theFace.Sf ⋅ U_b
#             convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
#             # diffusion 
#             diffusion = nu[theFace.iOwner] * theFace.gDiff
#             setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
#             # RHS/Source
#             RHS[theFace.iOwner] -= convection[1] - diffusion
#             RHS[theFace.iOwner+nCells] -= convection[2] -diffusion
#             RHS[theFace.iOwner+nCells+nCells] -= convection[3] - diffusion
#         end
#     end
#     return rows, cols, vals, RHS
# end # function batchedFaceBasedAssembly

function DivOnlyPrecalculatedWeightsUpwindFaceBasedAssembly(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    weights = input.weightsUpwind
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(P, nCells * 3)
    offsets = input.offsets
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = weights[iFace]                      # get precalculated weight

        valueUpper::P = ϕf * weights_f
        valueLower::P = -ϕf * (1 - weights_f)
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner])
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor])
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner)
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

function DivOnlyPrecalculatedWeightsCDFFaceBasedAssembly(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    weights = input.weightsCdf
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(P, nCells * 3)
    offsets = input.offsets
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = weights[iFace]                      # get precalculated weight

        valueUpper::P = ϕf * weights_f
        valueLower::P = -ϕf * (1 - weights_f)
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner])
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor])
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner)
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

function DivOnlyHardcodedUpwindFaceBasedAssembly(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(P, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    vals = zeros(P, entriesNeeded)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = upwind(ϕf)                          # get weight of transport variable interpolation 
                                                        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::P = weights_f * ϕf
        valueLower::P = -ϕf * (1 - weights_f)
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner])
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor])
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner)
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

function DivOnlyHardcodedCDFFaceBasedAssembly(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(P, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)

    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = 0.5                                 # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::P = weights_f * ϕf
        valueLower::P = -ϕf * (1 - weights_f)
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner])
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor])
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner)
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

function DivOnlyDynamicCDFFaceBasedAssembly(input::MatrixAssemblyInput, div::Function)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(P, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)

    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = div(ϕf)                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::P = ϕf * weights_f
        valueLower::P = -ϕf * (1 - weights_f)
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner])
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor])
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner)
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

function DivOnlyDynamicUpwindFaceBasedAssembly(input::MatrixAssemblyInput, div::Function)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(P, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = div(ϕf)                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::P = ϕf * weights_f
        valueLower::P = -ϕf * (1 - weights_f)
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner])
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor])
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner)
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

function LaplaceOnlyFaceBasedAssembly(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    weights = input.weightsCdf
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(P, nCells * 3)
    offsets = input.offsets
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor
        diffusion::P = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper::P = diffusion
        valueLower::P = -diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner])
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor])
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner)
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

function PrecalculatedWeightsCDFFaceBasedAssembly(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    weights = input.weightsCdf
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(P, nCells * 3)
    offsets = input.offsets
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = weights[iFace]                      # get precalculated weight

        valueUpper::P = ϕf * weights_f
        valueLower::P = -ϕf * (1 - weights_f)
        diffusion::P = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper::P += diffusion
        valueLower::P -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner])
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor])
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner)
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

function PrecalculatedWeightsUpwindFaceBasedAssembly(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    weights = input.weightsUpwind
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(P, nCells * 3)
    offsets = input.offsets
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = weights[iFace]                      # get precalculated weight

        valueUpper::P = ϕf * weights_f
        valueLower::P = -ϕf * (1 - weights_f)
        diffusion::P = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper::P += diffusion
        valueLower::P -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner])
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor])
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner)
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

function HardcodedUpwindFaceBasedAssembly(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary::Vector{BoundaryField{P}} = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(P, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    vals = zeros(P, entriesNeeded)
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
        if iFace < 3
            println(valueUpper, valueLower)
            
        end
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner])
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor])
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner)
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

function HardcodedCDFFaceBasedAssembly(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(P, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)

    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = 0.5                                 # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::P = weights_f * ϕf
        valueLower::P = -ϕf * (1 - weights_f)
        diffusion::P = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper::P += diffusion
        valueLower::P -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner])
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor])
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner)
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

function DynamicUpwindFaceBasedAssembly(input::MatrixAssemblyInput, div::Function)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(P, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = div(ϕf)                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::P = ϕf * weights_f
        valueLower::P = -ϕf * (1 - weights_f)
        diffusion::P = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper::P += diffusion
        valueLower::P -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner])
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor])
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner)
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

function DynamicCDFFaceBasedAssembly(input::MatrixAssemblyInput, div::Function)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(P, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)

    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = div(ϕf)                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::P = ϕf * weights_f
        valueLower::P = -ϕf * (1 - weights_f)
        diffusion::P = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper::P += diffusion
        valueLower::P -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner])
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor])
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner)
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