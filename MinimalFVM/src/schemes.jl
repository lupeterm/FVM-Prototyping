#### DIV SCHEMES
abstract type DivScheme{P} end

struct Noop{P} <: DivScheme{P} end

struct upwind{P} <: DivScheme{P} end
@inline (s::upwind{P})(ϕ) where {P<:AbstractFloat} = ifelse(ϕ ≥ 0, one(P), zero(P))

struct linear{P} <: DivScheme{P} end
@inline (u::linear{P})(ϕf) where {P<:AbstractFloat} = P(0.5)

#### DDT SCHEMES
abstract type DdtScheme{P} end

struct BDF1{P} <: DdtScheme{P} end
@inline (s::BDF1)(vol_c, Δt) = vol_c * Δt
