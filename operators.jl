

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