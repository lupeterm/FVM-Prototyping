include("init.jl")
include("cpu_helper.jl")
function FaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(Float32, nCells * 3)
    offsets, negOffsets, vals = getOffsetsAndValues(input)
    # offsets: the exact index the owner cell is placed in relation to neighbor indices,
    # negOffsets: where the cell faces start in relation to owner id
    # e.g: row values [1,2,101,10001,1,2,3...]
    #      offsets:   [1               2]
    #      negOffsets:[0               -1] := 1 is already correctly placed, 1 neighbor is smaller than 2 => -1
    # decrease negoffsets to walk from left to right
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    seenOwner = ones(Int32, nCells)

    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor
        fluxCn = nu[iOwner] * theFace.gDiff
        fluxFn = -fluxCn
        # set diag and upper
        setIndex!(iOwner, iOwner, fluxCn, rows, cols, vals, offsets[iOwner], entriesNeeded)
        setIndex!(iOwner, iNeighbor, fluxFn, rows, cols, vals, offsets[iOwner] + seenOwner[iOwner], entriesNeeded)
        # increment für symmetry
        seenOwner[iOwner] += 1
        # set diag and lower
        setIndex!(iNeighbor, iNeighbor, fluxCn, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        setIndex!(iNeighbor, iOwner, fluxFn, rows, cols, vals, offsets[iNeighbor] - negOffsets[iNeighbor], entriesNeeded)
        # increment für symmetry
        negOffsets[iNeighbor] -= 1
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
            fluxCn = nu[theFace.iOwner] * theFace.gDiff
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            @inbounds fluxVb = velocity_boundary[iBoundary].values[relativeFaceIndex] .* fluxCn
            setIndex!(theFace.iOwner, theFace.iOwner, fluxCn, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
            @inbounds RHS[theFace.iOwner] -= fluxVb[1]
            @inbounds RHS[theFace.iOwner+nCells] -= fluxVb[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= fluxVb[3]
        end
    end
    return rows, cols, vals, RHS
end # function batchedFaceBasedAssembly



function PrecalcPhiDivLapFaceBasedAssembly(input::MatrixAssemblyInput, weights::Vector{Float32})
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(Float32, nCells * 3)
    offsets::Vector{Int32}, negOffsets::Vector{Int32}, vals::Vector{Float32} = getOffsetsAndValues(input)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    seenOwner = ones(Int32, nCells)

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
        valueLower::Float32 = - ϕf * (1-weights_f)
        diffusion::Float32 = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper::Float32 += diffusion
        valueLower::Float32 -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner], entriesNeeded)
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] - negOffsets[iNeighbor], entriesNeeded)
      
        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + seenOwner[iOwner], entriesNeeded)  

        seenOwner[iOwner] += 1
        negOffsets[iNeighbor] -= 1
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
            ϕf  = theFace.Sf ⋅ U_b
            # diffusion 
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            # matrix[iOwner, iOwner] += ϕf - diffusion
            setIndex!(theFace.iOwner, theFace.iOwner, ϕf - diffusion, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
            # RHS/Source
            @inbounds fluxVb = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            @inbounds RHS[theFace.iOwner] -= fluxVb[1]
            @inbounds RHS[theFace.iOwner+nCells] -= fluxVb[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= fluxVb[3]
        end
    end
end

function HardcodedUpwindDivLapFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(Float32, nCells * 3)
    offsets::Vector{Int32}, negOffsets::Vector{Int32}, vals::Vector{Float32} = getOffsetsAndValues(input)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    seenOwner = ones(Int32, nCells)

    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor

        U_P = U[iOwner]     
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::Float32 = Uf ⋅ theFace.Sf                   # flux through the face
        weights_f = upwind(ϕf)                                 # get weight of transport variable interpolation 
                                                        # CDF -> 0.5, upwind -> ϕf >= 0 ? 1.0 : 0.0
        valueUpper::Float32 = 0.5ϕf
        valueLower::Float32 = - 0.5ϕf
        diffusion::Float32 = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper::Float32 += diffusion
        valueLower::Float32 -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner], entriesNeeded)
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] - negOffsets[iNeighbor], entriesNeeded)
      
        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + seenOwner[iOwner], entriesNeeded)  

        seenOwner[iOwner] += 1
        negOffsets[iNeighbor] -= 1
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
            ϕf  = theFace.Sf ⋅ U_b
            # diffusion 
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            # matrix[iOwner, iOwner] += ϕf - diffusion
            setIndex!(theFace.iOwner, theFace.iOwner, ϕf - diffusion, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
            # RHS/Source
            @inbounds fluxVb = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            @inbounds RHS[theFace.iOwner] -= fluxVb[1]
            @inbounds RHS[theFace.iOwner+nCells] -= fluxVb[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= fluxVb[3]
        end
    end
    return rows, cols, vals, RHS
end

function HardcodedCDFDivLapFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(Float32, nCells * 3)
    offsets::Vector{Int32}, negOffsets::Vector{Int32}, vals::Vector{Float32} = getOffsetsAndValues(input)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    seenOwner = ones(Int32, nCells)

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
        valueUpper::Float32 = 0.5ϕf
        valueLower::Float32 = - 0.5ϕf
        diffusion::Float32 = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper::Float32 += diffusion
        valueLower::Float32 -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner], entriesNeeded)
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] - negOffsets[iNeighbor], entriesNeeded)
      
        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + seenOwner[iOwner], entriesNeeded)  

        seenOwner[iOwner] += 1
        negOffsets[iNeighbor] -= 1
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
            ϕf  = theFace.Sf ⋅ U_b
            # diffusion 
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            # matrix[iOwner, iOwner] += ϕf - diffusion
            setIndex!(theFace.iOwner, theFace.iOwner, ϕf - diffusion, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
            # RHS/Source
            @inbounds fluxVb = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            @inbounds RHS[theFace.iOwner] -= fluxVb[1]
            @inbounds RHS[theFace.iOwner+nCells] -= fluxVb[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= fluxVb[3]
        end
    end
    return rows, cols, vals, RHS
end

function DynamicOnlineDivLapFaceBasedAssembly(input::MatrixAssemblyInput, div::Function)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(Float32, nCells * 3)
    offsets::Vector{Int32}, negOffsets::Vector{Int32}, vals::Vector{Float32} = getOffsetsAndValues(input)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    seenOwner = ones(Int32, nCells)

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
        valueLower::Float32 = - ϕf * (1-weights_f)
        diffusion::Float32 = nu[iOwner] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
        valueUpper::Float32 += diffusion
        valueLower::Float32 -= diffusion
        # contribution upper
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner], entriesNeeded)
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, valueUpper, rows, cols, vals, offsets[iNeighbor] - negOffsets[iNeighbor], entriesNeeded)
      
        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, valueLower, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, valueLower, rows, cols, vals, offsets[iOwner] + seenOwner[iOwner], entriesNeeded)  

        seenOwner[iOwner] += 1
        negOffsets[iNeighbor] -= 1
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
            ϕf  = theFace.Sf ⋅ U_b
            # diffusion 
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            # matrix[iOwner, iOwner] += ϕf - diffusion
            setIndex!(theFace.iOwner, theFace.iOwner, ϕf - diffusion, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
            # RHS/Source
            @inbounds fluxVb = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            @inbounds RHS[theFace.iOwner] -= fluxVb[1]
            @inbounds RHS[theFace.iOwner+nCells] -= fluxVb[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= fluxVb[3]
        end
    end
    return rows, cols, vals, RHS
end # function batchedFaceBasedAssembly