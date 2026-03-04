include("init.jl")

function JointFaceBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu::Vector{Float32} = input.nu
    velocity_boundary = input.U[1]
    nCells = length(mesh.cells)
    RHS = zeros(Float32, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces

    offsets, negativeOffsets, vals = getOffsetsAndValues(input)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    seenOwner = ones(Int32, nCells)
    @inbounds for iFace in eachindex(mesh.faces)
        theFace::Face = mesh.faces[iFace]
        iOwner = theFace.iOwner
        fluxCn = nu[iOwner] * theFace.gDiff
        if theFace.iNeighbor > 0
            iNeighbor = theFace.iNeighbor
            fluxFn = -fluxCn
            # set diag and upper
            setIndex!(iOwner, iOwner, fluxCn, rows, cols, vals, offsets[iOwner], entriesNeeded)
            setIndex!(iOwner, iNeighbor, fluxFn, rows, cols, vals, offsets[iOwner] + seenOwner[iOwner], entriesNeeded)
            # increment für symmetry
            seenOwner[iOwner] += 1
            # set diag and lower
            setIndex!(iNeighbor, iNeighbor, fluxCn, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
            setIndex!(iNeighbor, iOwner, fluxFn, rows, cols, vals, offsets[iNeighbor] - negativeOffsets[iNeighbor], entriesNeeded)
            # increment für symmetry
            negativeOffsets[iNeighbor] -= 1
        else
            iBoundary = theFace.patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = theFace.index - mesh.boundaries[iBoundary].startFace
            fluxVb::Vector{Float32} = velocity_boundary[iBoundary].values[relativeFaceIndex] .* -fluxCn
            idx = offsets[theFace.iOwner]
            setIndex!(iOwner, iOwner, fluxCn, rows, cols, vals, idx, entriesNeeded)
            RHS[iOwner] -= fluxVb[1]
            RHS[iOwner+nCells] -= fluxVb[2]
            RHS[iOwner+nCells+nCells] -= fluxVb[3]
        end
    end
    return rows, cols, vals, RHS
end # function faceBasedAssembly
