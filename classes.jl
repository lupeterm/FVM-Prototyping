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



abstract type DivScheme{P} end

struct Fused end
struct Noop{P} <: DivScheme{P} end

# function (n::Noop)(U_c::Vec3, U_n::Vec3, Sf::MVec3, nu::P, gdiff::P) 
#     diffusion = nu * gdiff
#     return -diffusion, diffusion
# end
struct UpwindScheme{P} <: DivScheme{P} end
# @inline function (s::UpwindScheme{P})(ϕf::P) where {P<:AbstractFloat}
#     ifelse(ϕf >= zero(P), one(P), zero(P))
# end
@inline (s::UpwindScheme)(ϕ) = ifelse(ϕ ≥ 0, 1, 0)

struct CentralDiffScheme{P} <: DivScheme{P} end
(u::CentralDiffScheme{T})(ϕf::T) where {T<:AbstractFloat} = 0.5

abstract type FVMOP{P<:AbstractFloat} end

struct LAPLACE{P<:AbstractFloat} <: FVMOP{P}
    operatorScaling::P
end

struct DIV{P<:AbstractFloat,S<:DivScheme{P}} <: FVMOP{P}
    scheme::S
    operatorScaling::P
end


function (d::DIV{P})(U_c::SVector{3,P}, U_n::SVector{3,P}, Sf::SVector{3,P}, nu::P, gdiff::P, retVals::MVector{2,P}) where {P<:AbstractFloat}
    Uf = 0.5(U_c + U_n)                  # interpolate velocity to face 
    ϕf = dot(Uf, Sf)                    # flux through the face
    weights_f = d.scheme(ϕf)                              # get weight of transport variable interpolation 
    retVals[1] += ϕf * weights_f
    retVals[2] += -ϕf * (1 - weights_f)  # valueupper, valuelower
end


function (t::LAPLACE{P})(U_c::SVector{3,P}, U_n::SVector{3,P}, Sf::SVector{3,P}, nu::P, gdiff::P, retVals::MVector{2,P}) where {P<:AbstractFloat}
    diffusion = nu * gdiff
    retVals[1] += -diffusion
    retVals[2] += diffusion
end



struct _DIV{P,S}
    scheme::S
    scale::P
end

struct _LAP{P}
    scale::P
end


function (d::_DIV{P,S})(
    U_c::SVector{3,P},
    U_n::SVector{3,P},
    Sf::SVector{3,P},
    nu::P,
    gdiff::P,
    a1::P,
    a2::P
) where {P<:AbstractFloat,S}
    Uf = 0.5(U_c + U_n)
    ϕf = dot(Uf, Sf)
    weights_f = d.scheme(ϕf)
    a1 += ϕf * weights_f
    a2 += -ϕf * (1 - weights_f)
    return a1, a2
end


function (t::_LAP{P})(
    U_c::SVector{3,P},
    U_n::SVector{3,P},
    Sf::SVector{3,P},
    nu::P,
    gdiff::P,
    a1::P,
    a2::P
) where {P<:AbstractFloat}
    diffusion = nu * gdiff
    a1 += -diffusion
    a2 += diffusion
    return a1, a2
end


struct OpSum{A,B}
    a::A
    b::B
end
@inline function (o::OpSum)(U_c, U_n, Sf, nu, gdiff, a1, a2)
    a1, a2 = o.a(U_c, U_n, Sf, nu, gdiff, a1, a2)
    a1, a2 = o.b(U_c, U_n, Sf, nu, gdiff, a1, a2)
    return a1, a2
end

Base.:+(a, b) = OpSum(a, b)
function runkernel(
    iOwners::CuArray{Int32},
    iNeighbors::CuArray{Int32},
    gDiffs::CuArray{P},
    offsets::CuArray{Int32},
    nu_g::CuArray{P},
    rows::CuArray{Int32},
    cols::CuArray{Int32},
    vals::CuArray{P},
    entriesNeeded::Int32,
    relativeToOwners::CuArray{Int32},
    N::Int32,
    relativeToNbs::CuArray{Int32},
    bFaceValues::CuArray{SVector{3,P}},
    RHS::CuArray{P},
    nCells::Int32,
    numBoundaryFaces::Int32,
    U::CuArray{SVector{3,P}},
    Sf::CuArray{SVector{3,P}},
    bFaceMapping::CuArray{Int32},
    ops,
) where {P<:AbstractFloat}
    backend = CUDABackend()
    fusedGpuAssembly(backend, 256)(
        iOwners,
        iNeighbors,
        gDiffs,
        offsets,
        nu_g,
        rows,
        cols,
        vals,
        RHS,
        entriesNeeded,
        relativeToOwners,
        N,
        relativeToNbs,
        U,
        Sf,
        bFaceMapping,
        bFaceValues,
        numBoundaryFaces,
        nCells,
        ops;
        ndrange=N + numBoundaryFaces
    )
    KernelAbstractions.synchronize(backend)
    return rows, cols, vals, RHS
end
using Atomix
@kernel function fusedGpuAssembly(
    @Const(iOwners),
    @Const(iNeighbors),
    @Const(gDiffs),
    @Const(offsets),
    @Const(nus),
    rows,
    cols,
    vals,
    RHS,
    @Const(entriesNeeded),
    @Const(relativeToOwner),
    @Const(numInteriorFaces),
    @Const(relativeToNeighbor),
    @Const(U),
    @Const(Sf),
    @Const(bFaceMapping),
    @Const(bFaceValues),
    @Const(numBoundaryFaces),
    @Const(nCells),
    ops,
)
    iFace = @index(Global)
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    if iFace <= numInteriorFaces

        upper = 0.0
        lower = 0.0

        upper, lower = ops(U[iOwner], U[iNeighbor], Sf[iFace], one(Float64), one(Float64), upper, lower)

        idx = offsets[iOwner]
        cols[idx] = iOwner
        rows[idx] = iOwner
        Atomix.@atomic vals[idx] += upper    # x

        idx = offsets[iOwner] + relativeToOwner[iFace]
        cols[idx] = iOwner
        rows[idx] = iNeighbor
        vals[idx] += lower    # x

        idx = offsets[iNeighbor]
        cols[idx] = iNeighbor
        rows[idx] = iNeighbor
        Atomix.@atomic vals[idx] += lower    # x

        idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
        cols[idx] = iNeighbor
        rows[idx] = iOwner
        vals[idx] += upper   # x
    else
        relativeFaceIndex = iFace - numInteriorFaces
        bFaceIndex = bFaceMapping[relativeFaceIndex]
        if bFaceIndex != -1
            convection = bFaceValues[bFaceIndex] .* dot(Sf[iFace], bFaceValues[bFaceIndex])
            # RHS/Source
            Atomix.@atomic RHS[iOwner] -= convection[1]
            Atomix.@atomic RHS[iOwner+nCells] -= convection[2]
            Atomix.@atomic RHS[iOwner+nCells+nCells] -= convection[3]
        end
    end
end


@kernel function fusedGpuAssemblyMin(
    @Const(ops),
    @Const(Uc),
    @Const(Un),
    @Const(Sf)
)
    i = @index(Global)

    ret = MVector{2,Float64}(zero(Float64), zero(Float64))

    ops(Uc, Un, Sf, one(Float64), one(Float64), ret)
end

