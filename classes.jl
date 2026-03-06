using StaticArrays

struct CaseDirError <: Exception
    message::String
end # struct CaseDirError

mutable struct Face
    index::Int32
    iNodes::Vector{Int32}
    iOwner::Int32
    iNeighbor::Int32
    centroid::MVector{3,Float32}
    Sf::MVector{3,Float32}
    area::Float32
    gDiff::Float32
    patchIndex::Int32
    relativeToOwner::Int32
    relativeToNeighbor::Int32
    batchId::Int32
end # struct Face

struct Boundary
    name::String
    type::String
    nFaces::Int32
    startFace::Int32
    index::Int32
end # struct Boundary

mutable struct Cell
    index::Int32
    nInternalFaces::Int32
    volume::Float32
    iFaces::Vector{Int32}
    iNeighbors::Vector{Int32}
    faceSigns::Vector{Int32}
    centroid::MVector{3,Float32}
    # iNodes::Vector{Int32}
    # numNeighbors::Int32
    # oldVolume::Float32
end # struct Cell

struct Mesh
    nodes::Vector
    faces::Vector{Face}
    boundaries::Vector{Boundary}
    numCells::Int32
    cells::Vector{Cell}
    numInteriorFaces::Int32
    numBoundaryCells::Int32
    numBoundaryFaces::Int32
end # struct Mesh

mutable struct Field
    values::Vector{SVector{3,Float32}}
end # struct Field

mutable struct BoundaryField
    name::String
    nFaces::Int32
    values::Vector{SVector{3,Float32}}
    type::String
end # struct BoundaryField

struct MatrixAssemblyInput
    mesh::Mesh
    nu::Vector{Float32}
    U::Tuple{Vector{BoundaryField},Field}
    weightsUpwind::Vector{Float32}
    weightsCdf::Vector{Float32}
    offsets::Vector{Int32}
    negOffsets::Vector{Int32}
end