include("init.jl")

function getOffsets(input::MatrixAssemblyInput)
    mesh = input.mesh
    neededEntries = length(mesh.cells) + 2 * mesh.numInteriorFaces
    nCells = length(mesh.cells)

    offsets::Vector{Int32} = ones(nCells + 1)
    for iElement in 2:nCells
        offsets[iElement] += offsets[iElement-1] + mesh.cells[iElement-1].nInternalFaces
    end
    offsets[end] = neededEntries + 1
    return offsets
end


@inline function setIndex!(ic::Int32, ir::Int32, val::P, rows::Vector{Int32}, cols::Vector{Int32}, vals::Vector{P}, idx::Int32) where {P<:AbstractFloat}
    cols[idx] = ic
    rows[idx] = ir
    vals[idx] += val   
end
