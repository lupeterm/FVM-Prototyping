include("init.jl")
include("cpu_helper.jl")

Base.:+(a::Tuple{P, P}, b::Tuple{P, P}) where {P<:AbstractFloat} = (a[1]+b[1], a[2]+b[2])

function prep(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    mesh = input.mesh
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(P, nCells * 3)
    offsets::Vector{Int32} = input.offsets
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    return rows, vals, cols, RHS, offsets
end
function FusedFaceBasedAssembly(input::MatrixAssemblyInput{P},rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}, ops::Vector{FVMOP{P}}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary::Vector{BoundaryField{P}} = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    # entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    # RHS = zeros(P, nCells * 3)
    # offsets::Vector{Int32} = input.offsets
    # vals = zeros(P, entriesNeeded)
    # rows = zeros(Int32, entriesNeeded)
    # cols = zeros(Int32, entriesNeeded)
    values = MVector{2, P}(zero(P), zero(P))
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor
        values .= 0
        for op in ops
            op(U[iOwner], U[iNeighbor], theFace.Sf, nu[iOwner], theFace.gDiff, values)
        end
        valueUpper = values[1]
        valueLower = values[2]
        # println("type: $(typeof(valueUpper))")
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        # println("type: $(typeof(valueUpper))")
        setIndex!(iOwner, iOwner, valueUpper, rows, cols, vals, offsets[iOwner])
        # return
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
            @inbounds convection = U_b .* ϕf
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
end # function batchedFaceBasedAssembly


function f(op, args...)
    u = 0.0
    l = 0.0
    for o in op
        a, b = o(args...)
        l += a
        b += b
    end
    return u, l
end

function ff(ops)
    args = (Vec3(1.0, 1.0, 1.0), Vec3(34.41, 12.3, 123.3), MVec3(1.0, 1.0, 1.0), 0.1, 4.3)
    [f(ops,args...)  for _ in 1:1000000]
end


function FusedFaceBasedAssembly2(input::MatrixAssemblyInput, ops::Vector{Function})
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
        valueUpper = 0.0
        valueLower = 0.0
        # println("type: $(typeof(valueUpper))")

        # for op::Function in ops
        #     values .+= op(U[iOwner], U[iNeighbor], theFace.Sf, nu[iOwner], theFace.gDiff)
        #     # valueUpper += op.valueUpper
        #     # valueLower += op.valueLower
        #     # println("u: $(typeof(u))")
        #     # println("l: $(typeof(l))")

        #     # valueUpper += u
        #     # valueLower += l
        # end
        # rowOwnStart + diagOffs[own]  -> (owner, owner)
        # println("type: $(typeof(valueUpper))")
        setIndex!(iOwner, iOwner, values[1], rows, cols, vals, offsets[iOwner])
        # return
        # rowNeiStart + neiOffs[facei] -> (neigh, owner)
        setIndex!(iNeighbor, iOwner, values[1], rows, cols, vals, offsets[iNeighbor] + theFace.relativeToNeighbor)

        # contribution lower
        # rowNeiStart + diagOffs[nei]  -> (neigh, neigh)
        setIndex!(iNeighbor, iNeighbor, values[2], rows, cols, vals, offsets[iNeighbor])
        # rowOwnStart + ownOffs[facei] -> (owner, neigh)
        setIndex!(iOwner, iNeighbor, values[2], rows, cols, vals, offsets[iOwner] + theFace.relativeToOwner)
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
