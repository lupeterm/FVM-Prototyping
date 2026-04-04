using StaticArrays
using KernelAbstractions
using Adapt
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
    iOwner::Int32
    iNeighbor::Int32
    centroid::MVector{3,P}
    Sf::SVector{3,P}
    area::P
    gDiff::P
    patchIndex::Int32
    ownerIdx::Int32
    neighborIdx::Int32
    ownerRelOwnerIdx::Int32
    neighborRelNeighborIdx::Int32
    batchId::Int32
end # struct Face

struct SOAFaces{P}
    iOwner::Vector{Int32}
    iNeighbor::Vector{Int32}
    Sf::Vector{SVector{3,P}}
    gDiff::Vector{P}
    batchId::Vector{Int32}
    ownerIdx::Vector{Int32}
    neighborIdx::Vector{Int32}
    ownerRelOwnerIdx::Vector{Int32}
    neighborRelNeighborIdx::Vector{Int32}
    patchIndex::Vector{Int32}
end



struct GPUSOAFaces{P<:AbstractFloat}
    iOwner::CuArray{Int32}
    iNeighbor::CuArray{Int32}
    Sf::CuArray{SVector{3,P}}
    gDiff::CuArray{P}
    batchId::CuArray{Int32}
    ownerIdx::CuArray{Int32}
    neighborIdx::CuArray{Int32}
    ownerRelOwnerIdx::CuArray{Int32}
    neighborRelNeighborIdx::CuArray{Int32}
    patchIndex::CuArray{Int32}
end



struct GpuFace{P}
    index::Int32
    iOwner::Int32
    iNeighbor::Int32
    Sf::SVector{3,P}
    area::P
    gDiff::P
    patchIndex::Int32
    relativeToOwner::Int32
    relativeToNeighbor::Int32
    batchId::Int32
end

(f::Face)() = GpuFace(
    f.index,
    f.iOwner,
    f.iNeighbor,
    f.Sf,
    f.area,
    f.gDiff,
    f.patchIndex,
    f.relativeToOwner,
    f.relativeToNeighbor,
    f.batchId
)

struct Boundary
    name::String
    type::String
    nFaces::Int32
    startFace::Int32
    index::Int32
end # struct Boundary

struct GpuBoundary
    isFixedValue::Bool
    nFaces::Int32
    startFace::Int32
    index::Int32
end
(b::Boundary)() = GpuBoundary(b.type=="fixedValue", b.nFaces, b.startFace, b.index)

mutable struct Cell{P<:AbstractFloat}
    index::Int32
    nInternalFaces::Int32
    volume::P
    iFaces::Vector{Int32}
    iNeighbors::Vector{Int32}
    faceSigns::Vector{Int32}
    centroid::MVector{3,P}
    rowOffset::Int32
    # iNodes::Vector{Int32}
    # numNeighbors::Int32
    # oldVolume::P
end # struct Cell

struct SOACells
    index::Vector{Int32}
    nInternalFaces::Vector{Int32}
    iFaces::Vector{Vector{Int32}}
    iNeighbors::Vector{Vector{Int32}}
    rowOffset::Vector{Int32}
end # struct Cell

toSOACells(input) = SOACells(
    [c.index for c in input.mesh.cells],
    [c.nInternalFaces for c in input.mesh.cells],
    [c.iFaces for c in input.mesh.cells],
    [c.iNeighbors for c in input.mesh.cells],
    [c.rowOffset for c in input.mesh.cells],
)

struct GpuCell{P}
    index::Int32
    nInternalFaces::Int32
    volume::P
    iFaces::SVector{6,Int32}
    faceSigns::SVector{6,Int32}
    centroid::SVector{3,P}
    iNeighbors::SVector{6,Int32}
end
function (c::Cell{P})() where {P<:AbstractFloat}
    return GpuCell{P}(
        c.index,
        c.nInternalFaces,
        c.volume,
        SVector{6,Int32}([c.iFaces; fill(-1, 6 - length(c.iFaces))]),
        SVector{6,Int32}([c.faceSigns; fill(-1, 6 - length(c.faceSigns))]),
        c.centroid,
        SVector{6,Int32}([c.iNeighbors; fill(-1, 6 - length(c.iNeighbors))])
    )
end
mutable struct Mesh{P<:AbstractFloat}
    nodes::Vector{SVector{3,P}}
    faces::Vector{Face{P}}
    boundaries::Vector{Boundary}
    numCells::Int32
    cells::Vector{Cell{P}}
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


struct SOAMatrixAssemblyInput{P<:AbstractFloat}
    numInteriorFaces::Int32
    boundaries::Vector{Boundary}
    nu::Vector{P}
    U_boundary::Vector{BoundaryField{P}}
    U_internal::Vector{SVector{3,P}}
    faces::SOAFaces{P}
    cells::SOACells
end

function toSOAInput(input::MatrixAssemblyInput)
    return SOAMatrixAssemblyInput(
        input.mesh.numInteriorFaces,
        input.mesh.boundaries,
        input.nu,
        input.U_boundary,
        input.U_internal,
        toSOAs(input),
        toSOACells(input)
    )
end

toSOAs(input::MatrixAssemblyInput) = SOAFaces(
    [f.iOwner for f in input.mesh.faces],
    [f.iNeighbor for f in input.mesh.faces],
    [f.Sf for f in input.mesh.faces],
    [f.gDiff for f in input.mesh.faces],
    [f.batchId for f in input.mesh.faces],
    [f.ownerIdx for f in input.mesh.faces],
    [f.neighborIdx for f in input.mesh.faces],
    [f.ownerRelOwnerIdx for f in input.mesh.faces],
    [f.neighborRelNeighborIdx for f in input.mesh.faces],
    [f.patchIndex for f in input.mesh.faces],
)


VectorToSOAs(faces::Vector{Face{P}}) where {P<:AbstractFloat} = SOAFaces(
    [f.iOwner for f in faces],
    [f.iNeighbor for f in faces],
    [f.Sf for f in faces],
    [f.gDiff for f in faces],
    [f.batchId for f in faces],
    [f.ownerIdx for f in faces],
    [f.neighborIdx for f in faces],
    [f.ownerRelOwnerIdx for f in faces],
    [f.neighborRelNeighborIdx for f in faces],
    [f.patchIndex for f in faces],
)

toGPUSOAs(soaFace::SOAFaces{P}) where{P<:AbstractFloat} = GPUSOAFaces{P}(
    soaFace.iOwner |> cu,
    soaFace.iNeighbor |> cu,
    soaFace.Sf |> cu,
    soaFace.gDiff |> cu,
    soaFace.batchId |> cu,
    soaFace.ownerIdx |> cu,
    soaFace.neighborIdx |> cu,
    soaFace.ownerRelOwnerIdx |> cu,
    soaFace.neighborRelNeighborIdx |> cu,
    soaFace.patchIndex |> cu,
)