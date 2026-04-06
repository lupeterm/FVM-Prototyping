include("init.jl")


function prepf(input::MatrixAssemblyInput{P}) where {P<:AbstractFloat}
    nCells = length(input.mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * input.mesh.numInteriorFaces
    RHS = zeros(P, nCells * 3)
    vals = zeros(P, entriesNeeded)
    return vals, RHS
end