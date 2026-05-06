module MinimalFVM

include("operators.jl")
using PrecompileTools

function test()
    println("test works")
end



function faceBased(
    numInteriorFaces::Int32,
    numCells::Int32,
    owner::Vector{Int32},
    neighbour::Vector{Int32},
    diagOffs::Vector{UInt8},
    ownOffs::Vector{UInt8},
    neiOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    phi_::Matrix{Float64},
    opString::String,
    faceFlux::Vector{Float64},
    gamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64}
)
    # phi = [Vector(phi_[:, i]) for i in axes(phi_, 2)]
    fused_pde = eval(Meta.parse(opString))
    for iFace in 1:numInteriorFaces
        iOwner = owner[iFace] + 1
        iNeighbor = neighbour[iFace] + 1

        rowNeiStart = rowOffs[iNeighbor] + 1
        rowOwnStart = rowOffs[iOwner] + 1

        valueUpper, valueLower = fused_pde(faceFlux[iFace], gamma[iFace], deltaCoeffs[iFace], magFaceArea[iFace], 0.0, 0.0) 
        vals[rowNeiStart+neiOffs[iFace]] += valueUpper
        vals[rowOwnStart+diagOffs[iOwner]] -= valueUpper
        vals[rowOwnStart+ownOffs[iFace]] += valueLower
        vals[rowNeiStart+diagOffs[iNeighbor]] -= valueLower
    end
end

@compile_workload begin
    d = eval(Meta.parse("Div{Float64, upwind{Float64}}(upwind{Float64}(), 1.0)"))
    laplace = Laplace{Float64}(1.0)
    laplace(1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    divU = Div{Float64,upwind{Float64}}(upwind{Float64}(), 1.0)
    divU(1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    divULap = divU + laplace
    divL = Div{Float64,linear{Float64}}(linear{Float64}(), 1.0)
    divL(1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    divLLap = divL + laplace
    divLLap(1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    faceBased(
        Int32(1),
        Int32(1),
        zeros(Int32, 1),
        zeros(Int32, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(Int32, 1),
        zeros(Float64, 1),
        zeros(Float64, 1, 1),
        "Div{Float64, upwind}(upwind{Float64}(), 1.0)",
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
    )
    faceBased(
        Int32(1),
        Int32(1),
        zeros(Int32, 1),
        zeros(Int32, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(Int32, 1),
        zeros(Float64, 1),
        zeros(Float64, 1, 1),
        "Div{Float64, linear}(linear{Float64}(), 1.0)",
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
    )
    faceBased(
        Int32(1),
        Int32(1),
        zeros(Int32, 1),
        zeros(Int32, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(Int32, 1),
        zeros(Float64, 1),
        zeros(Float64, 1, 1),
        "Laplace{Float64}(1.0)",
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
    )
    faceBased(
        Int32(1),
        Int32(1),
        zeros(Int32, 1),
        zeros(Int32, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(Int32, 1),
        zeros(Float64, 1),
        zeros(Float64, 1, 1),
        "Div{Float64, linear}(linear{Float64}(), 1.0) + Laplace{Float64}(1.0)",
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
    )
        faceBased(
        Int32(1),
        Int32(1),
        zeros(Int32, 1),
        zeros(Int32, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(Int32, 1),
        zeros(Float64, 1),
        zeros(Float64, 1, 1),
        "Div{Float64, upwind{Float64}}(upwind{Float64}(), 1.0) + Laplace{Float64}(1.0)",
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
    )

end
export faceBased, Div, Noop, linear, upwind, BDF1, Laplace, test

end # module MinimalFVM
