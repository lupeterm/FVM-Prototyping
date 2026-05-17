include("schemes.jl")
#### Transient Term 
# ddt(ϕ)

struct Ddt{P,S}
    scheme::S
    scale::P
end

# cellbased
function (d::Ddt{P,S})(
    oldVector::Vector{P},
    oldOldVector::Vector{P},
    volume::P,
    valueDiag::P,
    valueRHSx::P,
    valueRHSy::P,
    valueRHSz::P
) where {P<:AbstractFloat,S}
    ret = d.scale * d.scheme(volume, oldVector, oldOldVector)
    return valueDiag + ret[1], valueRHSx, +ret[2][1], valueRHSy, +ret[2][2], valueRHSz + ret[2][3]
end
# cellbased inner 
function (d::Ddt{P,BDF1{P}})(
    oldVector::Vector{P},
    volume::P,
    valueDiag::P,
    valueRHSx::P,
    valueRHSy::P,
    valueRHSz::P
) where {P<:AbstractFloat}
    ret = d.scale * d.scheme(volume, oldVector)
    return valueDiag + ret[1], valueRHSx, +ret[2][1], valueRHSy, +ret[2][2], valueRHSz + ret[2][3]
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
    return valueUpper - faceFlux * weights_f * d.scale, valueLower + faceFlux * (1 - weights_f) * d.scale
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
    return valueDiag + flux * valFrac2 * d.scale, valueRHSx - v[1] * d.scale, valueRHSy - v[2] * d.scale, valueRHSz - v[3] * d.scale
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
    return valueUpper + diffusion * t.scale, valueLower + diffusion * t.scale
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
    valueMat = flux * valueFraction * deltaCoeffGlobal
    v = flux * (valueFraction * deltaCoeffGlobal .* refValue
                +
                (1.0 - valueFraction) .* refGradient)
    return valueDiag - valueMat * d.scale, valueRHSx - v[1] * d.scale, valueRHSy - v[2] * d.scale, valueRHSz - v[3] * d.scale
end

#### Operator Fusing
struct DiffEq{A,B}
    a::A
    b::B
end

@inline function (o::DiffEq)(faceFlux, gamma, deltaCoeffs, magFaceArea, valueUpper, valueLower)
    valueUpper, valueLower = o.a(faceFlux, gamma, deltaCoeffs, magFaceArea, valueUpper, valueLower)
    valueUpper, valueLower = o.b(faceFlux, gamma, deltaCoeffs, magFaceArea, valueUpper, valueLower)
    return valueUpper, valueLower
end


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
Base.:+(a::PTERM, _::Nothing) = a
Base.:+(_::Nothing, a::PTERM) = a
Base.:+(_::Nothing, _::Nothing) = nothing
Base.:+(a::Tuple, b::Tuple) = a[1] + b[1], a[2] + b[2]
Base.:*(a::P, b::Tuple{P}) where {P<:AbstractFloat} = a * b[1], a * b[2]
Base.:*(a::P, b::Tuple{P, Vector{P}}) where {P<:AbstractFloat} = a * b[1], a .* b[2]

hasTransient(t::DiffEq) = hasTransient(t.a) || hasTransient(t.b)
hasTransient(t::Ddt) = true
hasTransient(t::Union{Div,Laplace}) = false

splitTempSpat(t::DiffEq) = splitTempSpat(t.a) + splitTempSpat(t.b)
splitTempSpat(t::Ddt) = t, nothing
splitTempSpat(t::Union{Div,Laplace}) = nothing, t

# Base.show(io::IO, t::DiffEq) = print(io, print(t.a), print(t.b)) 
# Base.show(io::IO, t::Div) = print(io, "$(t.scale) * Div{P, $(t.scheme)}" ) 
# Base.show(io::IO, t::Ddt) = print(io, "$(t.scale) * Ddt{P, $(t.scheme)}" ) 
# Base.show(io::IO, t::Laplace) = print(io, "+ $(t.scale) * Laplace{P}" ) 