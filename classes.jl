using StaticArrays

struct CaseDirError <: Exception
    message::String
end # struct CaseDirError

mutable struct Face{P}
    index::Int32
    iNodes::Vector{Int32}
    iOwner::Int32
    iNeighbor::Int32
    centroid::MVector{3,P}
    Sf::MVector{3,P}
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

struct Mesh{P<:AbstractFloat}
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


using Adapt
abstract type DivScheme{P} end

struct Fused end
struct Noop{P} <: DivScheme{P} end

# function (n::Noop)(U_c::Vec3, U_n::Vec3, Sf::MVec3, nu::P, gdiff::P) 
#     diffusion = nu * gdiff
#     return -diffusion, diffusion
# end
struct UpwindScheme{P} <: DivScheme{P} end
# (u::UpwindScheme{P})(ϕf::P)::P where {P<:AbstractFloat} = ϕf >= 0 ? one(P) : zero(P)

(u::UpwindScheme{T})(ϕf::T) where {T<:AbstractFloat} =
    ifelse(ϕf >= 0, one(T), zero(T))

struct CentralDiffScheme{P} <: DivScheme{P} end
# (c::CentralDiffScheme{P})(ϕf::P)::P = 0.5
(u::CentralDiffScheme{T})(ϕf::T) where {T<:AbstractFloat} = 0.5

abstract type FVMOP{P<:AbstractFloat} end

struct LAPLACE{P<:AbstractFloat} <: FVMOP{P}
    operatorScaling::P
end

struct DIV{P<:AbstractFloat} <: FVMOP{P}
    scheme::DivScheme{P}
    operatorScaling::P
end


# struct Operator{P<:AbstractFloat}
#     scheme::DivScheme{P}
#     operatorScaling::P
# end

# @inline function (o::Operator{P})(U_c::SVector{3,P}, U_n::SVector{3,P}, Sf::MVector{3,P}, nu::P, gdiff::P)::Tuple{P, P} where {S<:Union{UpwindScheme, CentralDiffScheme}, P<:AbstractFloat}
#     Uf = 0.5(U_c + U_n)                  # interpolate velocity to face 
#     ϕf = dot(Uf, Sf)                    # flux through the face
#     weights_f = o.scheme(ϕf)                              # get weight of transport variable interpolation 
#     return ϕf * weights_f, -ϕf * (1 - weights_f)  # valueupper, valuelower
# end



# @inline function (o::Operator{P})(U_c::SVector{3,P}, U_n::SVector{3,P}, Sf::MVector{3,P}, nu::P, gdiff::P)::Tuple{P, P} where {S<:Noop, P<:AbstractFloat}
#     diffusion = nu * gdiff
#     return -diffusion, diffusion
# end

function (d::DIV{P})(U_c::SVector{3,P}, U_n::SVector{3,P}, Sf::MVector{3,P}, nu::P, gdiff::P, retVals::MVector{2, P}) where {P<:AbstractFloat}
    Uf = 0.5(U_c + U_n)                  # interpolate velocity to face 
    ϕf = dot(Uf, Sf)                    # flux through the face
    weights_f = d.scheme(ϕf)                              # get weight of transport variable interpolation 
    # return ϕf * weights_f ,-ϕf * (1 - weights_f)  # valueupper, valuelower
    retVals[1] += ϕf * weights_f 
    retVals[2] += -ϕf * (1 - weights_f)  # valueupper, valuelower
end


function (t::LAPLACE{P})(U_c::SVector{3,P}, U_n::SVector{3,P}, Sf::MVector{3,P}, nu::P, gdiff::P, retVals::MVector{2, P})  where {P<:AbstractFloat}
    diffusion = nu * gdiff
    # return -diffusion, diffusion
    retVals[1] += -diffusion
    retVals[2] += diffusion
end
