#### DIV SCHEMES
abstract type DivScheme{P} end

struct Noop{P} <: DivScheme{P} end

struct upwind{P} <: DivScheme{P} end
@inline (s::upwind{P})(ϕ) where {P<:AbstractFloat} = ifelse(ϕ ≥ 0, one(P), zero(P))

struct linear{P} <: DivScheme{P} end
@inline (u::linear{P})(ϕf) where {P<:AbstractFloat} = P(0.5)

#### DDT SCHEMES
abstract type DdtScheme{P} end

struct BDF1{P} <: DdtScheme{P}
    a0a1::P
end

function BDF1(Δt::P) where {P<:AbstractFloat}
    return BDF1{P}(1 / Δt)
end

@inline function (s::BDF1{P})(volume::P, oldVector::Vector{P}) where {P<:AbstractFloat}
    common = volume * s.a0a1
    return common, common .* oldVector # diag, rhs 
end
struct BDF2{P} <: DdtScheme{P}
    a0::P
    a1::P
    a2::P
end
DELTAT = nothing
@inline function (s::BDF2)(volume::P, oldVector::Vector{P}, oldOldVector::Vector{P}) where {P<:AbstractFloat}
    diag = volume * s.a0
    rhs = volume * s.a1 .* oldVector[celli] + volume * s.a2 .* oldOldVector[celli]
    return diag, rhs
end
function BDF2(Δt::P)::BDF2{P} where {P<:AbstractFloat}
    return BDF2{P}(1.5 / Δt, 2.0 / Δt, -0.5 / Δt)
end


function BDF2(d::Nothing)::BDF2{Float64}
    println("[BDF2] No Δt was passed to assemble.")
    return BDF2{Float64}(1.0, 1.0, 1.0)
end
function BDF1(d::Nothing)::BDF1{Float64}
    println("[BDF1] No Δt was passed to assemble.")
    return BDF1{Float64}(1.0)
end