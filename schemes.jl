#### DIV SCHEMES
abstract type DivScheme{P} end

struct Noop{P} <: DivScheme{P} end

struct UpwindScheme{P} <: DivScheme{P} end
@inline (s::UpwindScheme{P})(ϕ) where {P<:AbstractFloat} = ifelse(ϕ ≥ 0, one(P), zero(P))

struct CentralDiffScheme{P} <: DivScheme{P} end
@inline (u::CentralDiffScheme{P})(ϕf) where {P<:AbstractFloat} = P(0.5)

#### DDT SCHEMES
abstract type DdtScheme{P} end

struct BDF1{P} <: DdtScheme{P} end
@inline (s::BDF1)(vol_c, Δt) = vol_c * Δt
