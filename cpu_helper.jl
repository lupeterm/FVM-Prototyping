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

function getOffsetsAndValues(input::MatrixAssemblyInput)#::Tuple{Vector{Int32},Vector{Float32}}
    mesh = input.mesh
    velocity_internal = input.U[2].values
    entries = length(mesh.cells) + 2 * mesh.numInteriorFaces
    nCells = length(mesh.cells)
    vals = Vector{Float32}(undef, entries * 3)

    offsets::Vector{Int32} = ones(Int32, nCells)
    negOffsets::Vector{Int32} = zeros(Int32, nCells)

    vals[1] = velocity_internal[1][1]
    vals[1+entries] = velocity_internal[1][2]
    vals[1+entries+entries] = velocity_internal[1][3]
    for iElement in 2:nCells
        offsets[iElement] += offsets[iElement-1] + mesh.cells[iElement-1].nInternalFaces
        vals[iElement] = velocity_internal[iElement][1]
        vals[iElement+entries] = velocity_internal[iElement][2]
        vals[iElement+entries+entries] = velocity_internal[iElement][3]
    end
    for iElement in 1:nCells
        theElement = mesh.cells[iElement]
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor > iElement
                negOffsets[theFace.iNeighbor] += 1
            end
        end
        offsets[iElement] += negOffsets[iElement]  # increase offset
    end
    return offsets, negOffsets, vals
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
