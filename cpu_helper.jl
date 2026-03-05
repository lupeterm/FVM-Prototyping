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


function setIndex!(ic::Int32, ir::Int32, val::Float32, rows::Vector{Int32}, cols::Vector{Int32}, vals::Vector{Float32}, idx::Int32, entries::Int32)
    # coeffMatrix[theFace.iOwner, theFace.iOwner] += fluxC
    cols[idx] = ic
    rows[idx] = ir
    vals[idx] += val    # x
    vals[idx+entries] += val  # y
    vals[idx+entries+entries] += val  # z
end

function setIndex!(ic::Int32, ir::Int32, val::MVector{3,Float32}, rows::Vector{Int32}, cols::Vector{Int32}, vals::Vector{Float32}, idx::Int32, entries::Int32)
    # coeffMatrix[theFace.iOwner, theFace.iOwner] += fluxC
    cols[idx] = ic
    rows[idx] = ir
    vals[idx] += val[1]    # x
    vals[idx+entries] += val[2]  # y
    vals[idx+entries+entries] += val[3]  # z
end
