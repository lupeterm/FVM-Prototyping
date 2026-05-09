include("schemes.jl")
#### Transient Term 
# ddt(ϕ)

struct Ddt{P,S}
    scheme::S
    scale::P
end

# cellbased inner 
function (d::Ddt{P,S})(
    # _::Vector{P},
    # _::Vector{P},
    _::Vector{P},
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
    faceFlux::P,
    _::P,
    _::P,
    _::P,
    valueUpper::P,
    valueLower::P
) where {P<:AbstractFloat,S}
    # ϕf = dot(0.5(U_c + U_n), Sf)     #### von NeoFoam/NeoN schon vorberechnet
    weights_f = d.scheme(faceFlux)
    return valueUpper - faceFlux * weights_f, valueLower + faceFlux * (1 - weights_f)
end


# facebased boundary 
function (d::Div{P,S})(
    refValue::Vector{P},
    refGradient::Vector{P},
    flux::P,
    valueFraction::P,
    deltaCoeffGlobal::P,
    _::P,
    _::P,
    _::P,
    valueDiag::P,
    valueRHSx::P,
    valueRHSy::P,
    valueRHSz::P
) where {P<:AbstractFloat,S}
    valFrac2 = 1.0 - valueFraction
    v = flux * valueFraction .* refValue + valFrac2 .* refGradient / deltaCoeffGlobal
    return valueDiag + flux * valFrac2, valueRHSx - v[1], valueRHSy - v[2], valueRHSz - v[3]
end

#### Diffusion
# laplacian(ν, ϕ)

struct Laplace{P}
    scale::P
end

function (t::Laplace{P})(
    _::P,
    gamma::P,
    deltaCoeffs::P,
    magFaceArea::P,
    valueUpper::P,
    valueLower::P
) where {P<:AbstractFloat}
    diffusion = gamma * deltaCoeffs * magFaceArea
    # valueUpper += -diffusion
    # valueLower += diffusion
    return valueUpper - diffusion, valueLower + diffusion
end

# facebased boundary 
function (d::Laplace{P})(
    refValue::Vector{P},
    refGradient::Vector{P},
    _::P,
    valueFraction::P,
    deltaCoeffGlobal::P,
    deltaCoeffBoundary::P,
    gamma::P,
    magFaceArea::P,
    valueDiag::P,
    valueRHSx::P,
    valueRHSy::P,
    valueRHSz::P
) where {P<:AbstractFloat}
    flux = gamma * magFaceArea
    valueMat = flux * valueFraction * deltaCoeffBoundary
    v = flux * (valueFraction * deltaCoeffBoundary .* refValue
                +
                (1.0 - valueFraction) .* refGradient)
    return valueDiag - valueMat, valueRHSx - v[1], valueRHSy - v[2], valueRHSz - v[3]
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
# @inline function (o::DiffEq)(U_c, U_n, faceFlux, gamma, deltaCoeffs, magFaceArea, valueUpper, valueLower)
@inline function (o::DiffEq)(faceFlux, gamma, deltaCoeffs, magFaceArea, valueUpper, valueLower)
    valueUpper, valueLower = o.a(faceFlux, gamma, deltaCoeffs, magFaceArea, valueUpper, valueLower)
    valueUpper, valueLower = o.b(faceFlux, gamma, deltaCoeffs, magFaceArea, valueUpper, valueLower)
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
@inline function (o::DiffEq)(
    refValue,
    refGradient,
    flux,
    valueFraction,
    deltaCoeffGlobal,
    deltaCoeffBoundary,
    gamma,
    magFaceArea,
    valueDiag,
    valueRHSx,
    valueRHSy,
    valueRHSz
)
    valueDiag, valueRHSx, valueRHSy, valueRHSz = o.a(refValue, refGradient, flux, valueFraction, deltaCoeffGlobal, deltaCoeffBoundary, gamma, magFaceArea, valueDiag, valueRHSx, valueRHSy, valueRHSz)
    valueDiag, valueRHSx, valueRHSy, valueRHSz = o.b(refValue, refGradient, flux, valueFraction, deltaCoeffGlobal, deltaCoeffBoundary, gamma, magFaceArea, valueDiag, valueRHSx, valueRHSy, valueRHSz)
    return valueDiag, valueRHSx, valueRHSy, valueRHSz
end

# cellbased inner 
# @inline function (DiffEq)(U_c, U_n, Sf, nu, gdiff, volume, dt, valueUpper, valueDiag, valueLower)
#     valueUpper, valueDiag, valueLower = o.a(U_c, U_n, Sf, nu, gdiff, volume, dt, valueUpper, valueDiag, valueLower)
#     valueUpper, valueDiag, valueLower = o.b(U_c, U_n, Sf, nu, gdiff, volume, dt, valueUpper, valueDiag, valueLower)
#     return valueUpper, valueDiag, valueLower
# end

PTERM = Union{DiffEq,Ddt,Div,Laplace}

Base.:+(a::PTERM, b::PTERM) = DiffEq(a, b)
