module Structs
export CaseDirError, Face, Node, Boundary, Cell, Mesh

struct CaseDirError <: Exception
    message::String
end # struct CaseDirError

mutable struct Face
    index::Int
    iNodes::Vector{Int}
    iOwner::Int
    iNeighbor::Int
    centroid::Vector{Float64}
    Sf::Vector{Float64}
    area::Float64
    CN::Vector{Float64}
    magCN::Float64
    eCN::Vector{Float64}
    gDiff::Float64
    T::Vector{Float64}
    gf::Float64
end # struct Face

struct Node
    centroid::Vector{Float64}
    faces::Vector{Face}
    iCells::Vector{Int}
    iFaces::Vector{Int}
end # struct Node

struct Boundary
    name::String
    type::String
    inGroups::Tuple{Int,String}
    nFaces::Int
    startFace::Int
end

mutable struct Cell
    index::Int
    iFaces::Vector{Int}
    neighbors::Vector{Int}
    numNeighbors::Int
    faceSigns::Vector{Int}
    iNodes::Vector{Int}
    volume::Float64
    oldVolume::Float64
    centroid::Vector{Float64}
end

struct Mesh
    nodes::Vector{Node}
    faces::Vector{Face}
    boundaries::Vector{Boundary}
    numCells::Int
    cells::Vector{Cell}
    numInteriorFaces::Int
    numBoundaryCells::Int
    numBoundaryFaces::Int
end

end # module Structs