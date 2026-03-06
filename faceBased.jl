include("init.jl")
include("cpu_helper.jl")

function _FaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(Float32, nCells * 3)
    offsets = input.offsets
    vals = zeros(Float32, entriesNeeded * 3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)


    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor
        diffusion = nu[iOwner] * theFace.gDiff
        fluxFn = -diffusion
        # set diag and upper
        setIndex!(iOwner, iOwner, diffusion, rows, cols, vals, offsets[iOwner], entriesNeeded)
        setIndex!(iOwner, iNeighbor, fluxFn, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner, entriesNeeded)
        # set diag and lower
        setIndex!(iNeighbor, iNeighbor, diffusion, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        setIndex!(iNeighbor, iOwner, fluxFn, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor, entriesNeeded)
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
            diffusion = U_b .* nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end # function batchedFaceBasedAssembly

function DivOnlyPrecalculatedWeightsUpwindFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    weights = input.weightsUpwind
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(Float32, nCells * 3)
    offsets = input.offsets
    vals = zeros(Float32, entriesNeeded * 3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::Float32 = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = weights[iFace]                      # get precalculated weight

        valueUpper::Float32 = ϕf * weights_f
        valueLower::Float32 = -ϕf * (1 - weights_f)
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner], entriesNeeded)
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor, entriesNeeded)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner, entriesNeeded)
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
            # diffusion = U_b .* nu[theFace.iOwner] * theFace.gDiff
            # setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
            # RHS/Source
            @inbounds RHS[theFace.iOwner] -= convection[1]
            @inbounds RHS[theFace.iOwner+nCells] -= convection[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= convection[3]
        end
    end
    return rows, cols, vals, RHS
end

function DivOnlyPrecalculatedWeightsCDFFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    weights = input.weightsCdf
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(Float32, nCells * 3)
    offsets = input.offsets
    vals = zeros(Float32, entriesNeeded * 3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::Float32 = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = weights[iFace]                      # get precalculated weight

        valueUpper::Float32 = ϕf * weights_f
        valueLower::Float32 = -ϕf * (1 - weights_f)
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner], entriesNeeded)
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor, entriesNeeded)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner, entriesNeeded)
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
            # diffusion = U_b .* nu[theFace.iOwner] * theFace.gDiff
            # setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
            # RHS/Source
            @inbounds RHS[theFace.iOwner] -= convection[1]
            @inbounds RHS[theFace.iOwner+nCells] -= convection[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= convection[3]
        end
    end
    return rows, cols, vals, RHS
end

function DivOnlyHardcodedUpwindFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(Float32, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    vals = zeros(Float32, entriesNeeded * 3)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::Float32 = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = upwind(ϕf)                          # get weight of transport variable interpolation 
                                                        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::Float32 = weights_f * ϕf
        valueLower::Float32 = -ϕf * (1 - weights_f)
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner], entriesNeeded)
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor, entriesNeeded)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner, entriesNeeded)
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
            # diffusion = U_b .* nu[theFace.iOwner] * theFace.gDiff
            # setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
            # RHS/Source
            @inbounds RHS[theFace.iOwner] -= convection[1]
            @inbounds RHS[theFace.iOwner+nCells] -= convection[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= convection[3]
        end
    end
    return rows, cols, vals, RHS
end

function DivOnlyHardcodedCDFFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(Float32, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    vals = zeros(Float32, entriesNeeded * 3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)

    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::Float32 = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = 0.5                                 # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::Float32 = weights_f * ϕf
        valueLower::Float32 = -ϕf * (1 - weights_f)
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner], entriesNeeded)
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor, entriesNeeded)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner, entriesNeeded)
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
            # diffusion = U_b .* nu[theFace.iOwner] * theFace.gDiff
            # setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
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
    RHS = zeros(Float32, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    vals = zeros(Float32, entriesNeeded * 3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)

    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::Float32 = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = div(ϕf)                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::Float32 = ϕf * weights_f
        valueLower::Float32 = -ϕf * (1 - weights_f)
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner], entriesNeeded)
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor, entriesNeeded)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner, entriesNeeded)
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
    RHS = zeros(Float32, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    vals = zeros(Float32, entriesNeeded * 3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::Float32 = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = div(ϕf)                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::Float32 = ϕf * weights_f
        valueLower::Float32 = -ϕf * (1 - weights_f)
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner], entriesNeeded)
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor, entriesNeeded)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner, entriesNeeded)
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
            # diffusion = U_b .* nu[theFace.iOwner] * theFace.gDiff
            # setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
            # RHS/Source
            @inbounds RHS[theFace.iOwner] -= convection[1]
            @inbounds RHS[theFace.iOwner+nCells] -= convection[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= convection[3]
        end
    end
    return rows, cols, vals, RHS
end

function LaplaceOnlyFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    weights = input.weightsCdf
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(Float32, nCells * 3)
    offsets = input.offsets
    vals = zeros(Float32, entriesNeeded * 3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor
        diffusion::Float32 = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper::Float32 = diffusion
        valueLower::Float32 = -diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner], entriesNeeded)
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor, entriesNeeded)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner, entriesNeeded)
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
            diffusion = U_b .* nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
            # RHS/Source
            @inbounds RHS[theFace.iOwner] -= diffusion[1]
            @inbounds RHS[theFace.iOwner+nCells] -= diffusion[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= diffusion[3]
        end
    end
    return rows, cols, vals, RHS
end

function PrecalculatedWeightsCDFFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    weights = input.weightsCdf
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(Float32, nCells * 3)
    offsets = input.offsets
    vals = zeros(Float32, entriesNeeded * 3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::Float32 = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = weights[iFace]                      # get precalculated weight

        valueUpper::Float32 = ϕf * weights_f
        valueLower::Float32 = -ϕf * (1 - weights_f)
        diffusion::Float32 = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper::Float32 += diffusion
        valueLower::Float32 -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner], entriesNeeded)
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor, entriesNeeded)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner, entriesNeeded)
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
            diffusion = U_b .* nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end

function PrecalculatedWeightsUpwindFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    weights = input.weightsUpwind
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(Float32, nCells * 3)
    offsets = input.offsets
    vals = zeros(Float32, entriesNeeded * 3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::Float32 = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = weights[iFace]                      # get precalculated weight

        valueUpper::Float32 = ϕf * weights_f
        valueLower::Float32 = -ϕf * (1 - weights_f)
        diffusion::Float32 = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper::Float32 += diffusion
        valueLower::Float32 -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner], entriesNeeded)
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor, entriesNeeded)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner, entriesNeeded)
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
            diffusion = U_b .* nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end

function HardcodedUpwindFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(Float32, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    vals = zeros(Float32, entriesNeeded * 3)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::Float32 = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = upwind(ϕf)                          # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::Float32 = weights_f * ϕf
        valueLower::Float32 = -ϕf * (1 - weights_f)
        diffusion::Float32 = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper::Float32 += diffusion
        valueLower::Float32 -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner], entriesNeeded)
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor, entriesNeeded)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner, entriesNeeded)
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
            diffusion = U_b .* nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end

function HardcodedCDFFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(Float32, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    vals = zeros(Float32, entriesNeeded * 3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)

    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::Float32 = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = 0.5                                 # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::Float32 = weights_f * ϕf
        valueLower::Float32 = -ϕf * (1 - weights_f)
        diffusion::Float32 = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper::Float32 += diffusion
        valueLower::Float32 -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner], entriesNeeded)
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor, entriesNeeded)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner, entriesNeeded)
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
            diffusion = U_b .* nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
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
    RHS = zeros(Float32, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    vals = zeros(Float32, entriesNeeded * 3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::Float32 = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = div(ϕf)                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::Float32 = ϕf * weights_f
        valueLower::Float32 = -ϕf * (1 - weights_f)
        diffusion::Float32 = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper::Float32 += diffusion
        valueLower::Float32 -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner], entriesNeeded)
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor, entriesNeeded)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner, entriesNeeded)
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
            diffusion = U_b .* nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
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
    RHS = zeros(Float32, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    vals = zeros(Float32, entriesNeeded * 3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)

    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::Float32 = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = div(ϕf)                             # get weight of transport variable interpolation 
        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::Float32 = ϕf * weights_f
        valueLower::Float32 = -ϕf * (1 - weights_f)
        diffusion::Float32 = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper::Float32 += diffusion
        valueLower::Float32 -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner], entriesNeeded)
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor, entriesNeeded)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner, entriesNeeded)
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
            diffusion = U_b .* nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end # function batchedFaceBasedAssembly