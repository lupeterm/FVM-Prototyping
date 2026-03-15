using StaticArrays
using KernelAbstractions
struct CaseDirError <: Exception
    message::String
end # struct CaseDirError


mutable struct TmpFace
    index::Int32
    iNodes::Vector{Int32}
    iOwner::Int32
    iNeighbor::Int32
end # struct TmpFace

mutable struct Face{P}
    index::Int32
    # iNodes::Vector{Int32}
    iOwner::Int32
    iNeighbor::Int32
    centroid::MVector{3,P}
    Sf::SVector{3,P}
    area::P
    gDiff::P
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

mutable struct Cell{P<:AbstractFloat}
    index::Int32
    nInternalFaces::Int32
    volume::P
    iFaces::Vector{Int32}
    iNeighbors::Vector{Int32}
    faceSigns::Vector{Int32}
    centroid::MVector{3,P}
    # iNodes::Vector{Int32}
    # numNeighbors::Int32
    # oldVolume::P
end # struct Cell

mutable struct Mesh{P<:AbstractFloat}
    nodes::Vector{SVector{3,P}}
    faces::Vector{Face{P}}
    boundaries::Vector{Boundary}
    numCells::Int32
    cells::Vector{Cell}
    numInteriorFaces::Int32
    numBoundaryCells::Int32
    numBoundaryFaces::Int32
end # struct Mesh

mutable struct Field{P<:AbstractFloat}
    values::Vector{SVector{3,P}}
end # struct Field

mutable struct BoundaryField{P<:AbstractFloat}
    name::String
    nFaces::Int32
    values::Vector{SVector{3,P}}
    type::String
end # struct BoundaryField

struct MatrixAssemblyInput{P<:AbstractFloat}
    mesh::Mesh{P}
    nu::Vector{P}
    U_boundary::Vector{BoundaryField{P}}
    U_internal::Vector{SVector{3,P}}
    weightsUpwind::Vector{P}
    weightsCdf::Vector{P}
    offsets::Vector{Int32}
    negOffsets::Vector{Int32}
end



