include("schemes.jl")
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
    ϕf::P = dot(Uf, Sf)
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
    ϕf = dot(U_b, Sf)
    conv = U_b .* ϕf
    valueRHSx -= conv[1]
    valueRHSy -= conv[2]
    valueRHSz -= conv[3] 
    return valueDiag, valueRHSx, valueRHSy, valueRHSz
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
    valueRHSx -= diffusion
    valueRHSy -= diffusion
    valueRHSz -= diffusion
    return valueDiag, valueRHSx, valueRHSy, valueRHSz  
end

#### Operator Fusing
struct DiffEq{A,B}
    a::A
    b::B
end

# facebased inner first call 
# @inline function (o::DiffEq)(U_c, U_n, Sf, nu, gdiff)
#     valueUpper, valueLower = o.a(U_c, U_n, Sf, nu, gdiff, zero(typeof(nu)), zero(typeof(nu)))
#     valueUpper, valueLower = o.b(U_c, U_n, Sf, nu, gdiff, valueUpper, valueLower)
#     return valueUpper, valueLower
# end
# facebased inner 
@inline function (o::DiffEq)(U_c, U_n, Sf, nu, gdiff, valueUpper, valueLower)
    valueUpper, valueLower = o.a(U_c, U_n, Sf, nu, gdiff, valueUpper, valueLower)
    valueUpper, valueLower = o.b(U_c, U_n, Sf, nu, gdiff, valueUpper, valueLower)
    return valueUpper, valueLower
end

# facebased boundary first call
# @inline function (o::DiffEq)(U_b, Sf, nu, gdiff)
#     t = typeof(nu)
#     diag, rhsx, rhsy, rhsz = o.a(U_b, Sf, nu, gdiff, zero(t), zero(t), zero(t), zero(t))
#     diag, rhsx, rhsy, rhsz = o.b(U_b, Sf, nu, gdiff, diag, rhsx, rhsy, rhsz)
#     return diag, rhsx, rhsy, rhsz
# end

# facebased boundary 
@inline function (o::DiffEq)(U_b, Sf, nu, gdiff, diag, rhsx, rhsy, rhsz)
    diag, rhsx, rhsy, rhsz = o.a(U_b, Sf, nu, gdiff, diag, rhsx, rhsy, rhsz)
    diag, rhsx, rhsy, rhsz = o.b(U_b, Sf, nu, gdiff, diag, rhsx, rhsy, rhsz)
    return diag, rhsx, rhsy, rhsz
end


# cellbased inner 
# @inline function (DiffEq)(U_c, U_n, Sf, nu, gdiff, volume, dt, valueUpper, valueDiag, valueLower)
#     valueUpper, valueDiag, valueLower = o.a(U_c, U_n, Sf, nu, gdiff, volume, dt, valueUpper, valueDiag, valueLower)
#     valueUpper, valueDiag, valueLower = o.b(U_c, U_n, Sf, nu, gdiff, volume, dt, valueUpper, valueDiag, valueLower)
#     return valueUpper, valueDiag, valueLower
# end

PTERM = Union{DiffEq,Ddt,Div,Laplace}

Base.:+(a::PTERM, b::PTERM) = DiffEq(a, b)
