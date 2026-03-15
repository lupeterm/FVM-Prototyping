abstract type DivScheme{P} end

struct Noop{P} <: DivScheme{P} end

struct UpwindScheme{P} <: DivScheme{P} end
@inline (s::UpwindScheme)(ϕ) = ifelse(ϕ ≥ 0, 1, 0)

struct CentralDiffScheme{P} <: DivScheme{P} end
@inline (u::CentralDiffScheme)(ϕf) = 0.5