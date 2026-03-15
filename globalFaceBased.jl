include("init.jl")
include("cpu_helper.jl")

 function GlobalFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu::Vector{P} = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    weights = input.weightsCdf
    nCells = length(mesh.cells)
    RHS = zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = input.offsets
    vals = zeros(P, entriesNeeded*3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in eachindex(mesh.faces)
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        diffusion = nu[iOwner] * theFace.gDiff
        if theFace.iNeighbor > 0
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
        else
            iBoundary = theFace.patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = theFace.index - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion .* U_b, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end # function GlobalFaceBasedAssembly

function LaplaceOnlyGlobalFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu::Vector{P} = input.nu
    velocity_boundary = input.U[1]
    nCells = length(mesh.cells)
    RHS = zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = input.offsets
    vals = zeros(P, entriesNeeded*3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in eachindex(mesh.faces)
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        diffusion = nu[iOwner] * theFace.gDiff
        if theFace.iNeighbor > 0
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
        else
            iBoundary = theFace.patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = theFace.index - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            # ϕf = theFace.Sf ⋅ U_b
            # @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion .* U_b, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            @inbounds RHS[theFace.iOwner] -= diffusion
            @inbounds RHS[theFace.iOwner+nCells] -= diffusion
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= diffusion
        end
    end
    return rows, cols, vals, RHS
end
function DivOnlyPrecalculatedWeightsUpwindGlobalFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu::Vector{P} = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values

    nCells = length(mesh.cells)
    weights = input.weightsUpwind
    RHS = zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = input.offsets
    vals = zeros(P, entriesNeeded*3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in eachindex(mesh.faces)
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        if theFace.iNeighbor > 0
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
        else
            iBoundary = theFace.patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
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
function DivOnlyPrecalculatedWeightsCDFGlobalFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu::Vector{P} = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values

    nCells = length(mesh.cells)
    weights = input.weightsCdf
    RHS = zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = input.offsets
    vals = zeros(P, entriesNeeded*3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in eachindex(mesh.faces)
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        if theFace.iNeighbor > 0
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
        else
            iBoundary = theFace.patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
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
function DivOnlyHardcodedUpwindGlobalFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu::Vector{P} = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values

    nCells = length(mesh.cells)
    RHS = zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = input.offsets
    vals = zeros(P, entriesNeeded*3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in eachindex(mesh.faces)
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        if theFace.iNeighbor > 0
            iNeighbor = theFace.iNeighbor
            U_P = U[iOwner]
            U_N = U[iNeighbor]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = upwind(ϕf)                      # get precalculated weight

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
        else
            iBoundary = theFace.patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
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
function DivOnlyHardcodedCDFGlobalFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu::Vector{P} = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    RHS = zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = input.offsets
    vals = zeros(P, entriesNeeded*3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in eachindex(mesh.faces)
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        if theFace.iNeighbor > 0
            iNeighbor = theFace.iNeighbor
            U_P = U[iOwner]
            U_N = U[iNeighbor]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = centralDifferencing(ϕf)                      # get precalculated weight

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
        else
            iBoundary = theFace.patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
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
function DivOnlyDynamicCDFGlobalFaceBasedAssembly(input::MatrixAssemblyInput, div::Function)
    mesh = input.mesh
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    RHS = zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = input.offsets
    vals = zeros(P, entriesNeeded*3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in eachindex(mesh.faces)
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        if theFace.iNeighbor > 0
            iNeighbor = theFace.iNeighbor
            U_P = U[iOwner]
            U_N = U[iNeighbor]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = div(ϕf)                      # get precalculated weight

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
        else
            iBoundary = theFace.patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
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
function DivOnlyDynamicUpwindGlobalFaceBasedAssembly(input::MatrixAssemblyInput, div::Function)
    mesh = input.mesh
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    RHS = zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = input.offsets
    vals = zeros(P, entriesNeeded*3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in eachindex(mesh.faces)
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        if theFace.iNeighbor > 0
            iNeighbor = theFace.iNeighbor
            U_P = U[iOwner]
            U_N = U[iNeighbor]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = div(ϕf)                             # get precalculated weight

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
        else
            iBoundary = theFace.patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
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
function PrecalculatedWeightsUpwindGlobalFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu::Vector{P} = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    weights = input.weightsUpwind
    RHS = zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = input.offsets
    vals = zeros(P, entriesNeeded*3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in eachindex(mesh.faces)
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        diffusion::P = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        if theFace.iNeighbor > 0
            iNeighbor = theFace.iNeighbor
                
            U_P = U[iOwner]
            U_N = U[iNeighbor]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = weights[iFace]                      # get precalculated weight

            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)
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
        else
            iBoundary = theFace.patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # RHS/Source
            value = convection .+ diffusion
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion .* U_b, rows, cols, vals, offsets[theFace.iOwner])
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end
function PrecalculatedWeightsCDFGlobalFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu::Vector{P} = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    weights = input.weightsCdf
    RHS = zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = input.offsets
    vals = zeros(P, entriesNeeded*3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in eachindex(mesh.faces)
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        diffusion::P = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        if theFace.iNeighbor > 0
            iNeighbor = theFace.iNeighbor
                
            U_P = U[iOwner]
            U_N = U[iNeighbor]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = weights[iFace]                      # get precalculated weight

            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)
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
        else
            iBoundary = theFace.patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # RHS/Source
            value = convection .+ diffusion
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion .* U_b, rows, cols, vals, offsets[theFace.iOwner])
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end
function HardcodedUpwindGlobalFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu::Vector{P} = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values

    nCells = length(mesh.cells)
    RHS = zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = input.offsets
    vals = zeros(P, entriesNeeded*3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in eachindex(mesh.faces)
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        diffusion::P = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        if theFace.iNeighbor > 0
            iNeighbor = theFace.iNeighbor
                
            U_P = U[iOwner]
            U_N = U[iNeighbor]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = upwind(ϕf)                      # get precalculated weight

            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)
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
        else
            iBoundary = theFace.patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # RHS/Source
            value = convection .+ diffusion
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion .* U_b, rows, cols, vals, offsets[theFace.iOwner])
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end
function HardcodedCDFGlobalFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu::Vector{P} = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values

    nCells = length(mesh.cells)
    RHS = zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = input.offsets
    vals = zeros(P, entriesNeeded*3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in eachindex(mesh.faces)
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        diffusion::P = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        if theFace.iNeighbor > 0
            iNeighbor = theFace.iNeighbor
                
            U_P = U[iOwner]
            U_N = U[iNeighbor]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = centralDifferencing(ϕf)                      # get precalculated weight

            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)
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
        else
            iBoundary = theFace.patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # RHS/Source
            value = convection .+ diffusion
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion .* U_b, rows, cols, vals, offsets[theFace.iOwner])
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end
function DynamicCDFGlobalFaceBasedAssembly(input::MatrixAssemblyInput, div::Function)
    mesh = input.mesh
    nu::Vector{P} = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values

    nCells = length(mesh.cells)
    RHS = zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = input.offsets
    vals = zeros(P, entriesNeeded*3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in eachindex(mesh.faces)
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        diffusion::P = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        if theFace.iNeighbor > 0
            iNeighbor = theFace.iNeighbor
                
            U_P = U[iOwner]
            U_N = U[iNeighbor]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = div(ϕf)                      # get precalculated weight

            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)
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
        else
            iBoundary = theFace.patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # RHS/Source
            value = convection .+ diffusion
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion .* U_b, rows, cols, vals, offsets[theFace.iOwner])
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end
function DynamicUpwindGlobalFaceBasedAssembly(input::MatrixAssemblyInput, div::Function)
    mesh = input.mesh
    nu::Vector{P} = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values

    nCells = length(mesh.cells)
    RHS = zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = input.offsets
    vals = zeros(P, entriesNeeded*3)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    @inbounds for iFace in eachindex(mesh.faces)
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        diffusion::P = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        if theFace.iNeighbor > 0
            iNeighbor = theFace.iNeighbor
                
            U_P = U[iOwner]
            U_N = U[iNeighbor]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = div(ϕf)                      # get precalculated weight

            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)
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
        else
            iBoundary = theFace.patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # RHS/Source
            value = convection .+ diffusion
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion .* U_b, rows, cols, vals, offsets[theFace.iOwner])
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
    end
    return rows, cols, vals, RHS
end