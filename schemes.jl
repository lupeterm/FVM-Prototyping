#### DIV SCHEMES
abstract type DivScheme{P} end

struct Noop{P} <: DivScheme{P} end

struct UpwindScheme{P} <: DivScheme{P} end
@inline (s::UpwindScheme)(ϕ) = ifelse(ϕ ≥ 0, 1, 0)

struct CentralDiffScheme{P} <: DivScheme{P} end
@inline (u::CentralDiffScheme)(ϕf) = 0.5

#### DDT SCHEMES
abstract type DdtScheme{P} end

struct BDF1{P} <: DdtScheme{P} end
@inline (s::BDF1)(vol_c, Δt) = vol_c * Δt
