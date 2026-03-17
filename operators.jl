#### Operator Fusing
struct DiffEq{A,B}
    a::A
    b::B
end
# facebased inner 
@inline function (DiffEq)(U_c, U_n, Sf, nu, gdiff, valueUpper, valueLower)
    valueUpper, valueLower = o.b(U_c, U_n, Sf, nu, gdiff, valueUpper, valueLower)
    valueUpper, valueLower = o.a(U_c, U_n, Sf, nu, gdiff, valueUpper, valueLower)
    return valueUpper, valueLower
end

# cellbased inner 
@inline function (DiffEq)(U_c, U_n, Sf, nu, gdiff, volume, dt, valueUpper, valueDiag, valueLower)
    valueUpper, valueDiag, valueLower = o.b(U_c, U_n, Sf, nu, gdiff, volume, dt, valueUpper, valueDiag, valueLower)
    valueUpper, valueDiag, valueLower = o.a(U_c, U_n, Sf, nu, gdiff, volume, dt, valueUpper, valueDiag, valueLower)
    return valueUpper, valueDiag, valueLower
end

PTERM = Union{DiffEq, Ddt, Div, Laplace}

Base.:+(a::PTERM, b::PTERM) = DiffEq(a, b)
    
#### Transient Term 
# ddt(ϕ)

struct Ddt{P,S}
    scheme::S
    scale::P
end

# cellbased inner 
function (d::Ddt{P,S})(
    _::SVector{3,P},
    _::SVector{3,P},
    _::SVector{3,P},
    _::P,
    _::P,
    volume::P,
    dt::P,
    valueUpper::P,
    valueDiag::P,
    valueLower::P
) where {P<:AbstractFloat,S}
    valueDiag += d.scheme(volume, dt)
    return valueUpper, valueDiag, valueLower
end

#### Convection
# ∇(ϕ,U) (in momentum equation)

struct Div{P,S}
    scheme::S
    scale::P
end

# facebased inner
function (d::Div{P,S})(
    U_c::SVector{3,P},
    U_n::SVector{3,P},
    Sf::SVector{3,P},
    _::P,
    _::P,
    valueUpper::P,
    valueLower::P
) where {P<:AbstractFloat,S}
    Uf = 0.5(U_c + U_n)
    ϕf = dot(Uf, Sf)
    weights_f = d.scheme(ϕf)
    valueUpper += ϕf * weights_f
    valueLower += -ϕf * (1 - weights_f)
    return valueUpper, valueLower
end


# facebased boundary 
function (d::Div{P,S})(
    U_b::SVector{3,P},
    Sf::SVector{3,P},
    _::P,
    _::P,
    valueDiag::P,
    valueRHSx::P,
    valueRHSy::P,
    valueRHSz::P
) where {P<:AbstractFloat,S}
    ϕf = dot(Uf, Sf)
    return valueDiag, (valueRHSx, valueRHSy, valueRHSz) .- U_b .* ϕf
end

#### Diffusion
# laplacian(ν, ϕ)

struct Laplace{P}
    scale::P
end

function (t::Laplace{P})(
    _::SVector{3,P},
    _::SVector{3,P},
    Sf::SVector{3,P},
    nu::P,
    gDiff::P,
    valueUpper::P,
    valueLower::P
) where {P<:AbstractFloat}
    diffusion = nu * gDiff
    valueUpper += -diffusion
    valueLower += diffusion
    return valueUpper, valueLower
end

# facebased boundary 
function (d::Laplace{P})(
    _::SVector{3,P},
    _::SVector{3,P},
    nu::P,
    gDiff::P,
    valueDiag::P,
    valueRHSx::P,
    valueRHSy::P,
    valueRHSz::P
) where {P<:AbstractFloat}
    diffusion = nu * gDiff
    valueDiag -= diffusion
    return valueDiag, (valueRHSx, valueRHSy, valueRHSz) .- diffusion
end